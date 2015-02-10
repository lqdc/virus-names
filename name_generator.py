"""Module responsible for loading trained models and guessing virus names.

Overall, the idea is to guess a name for a virus from the names that Antiviruses
give it.

When using the Guesser class, all Exceptions are wrapped in a
NameGeneratorException.

Usage:
In [1]: from name_generator import Guesser
In [2]: g = Guesser()
In [3]: g.guess_everything({"F-Prot": "W32/AddInstall.A",
   ...:                     "Comodo": "Application.Win32.InstalleRex.KG"})
Out[3]:
{'compiler': 'unknown',
 '_type': 'Application',
 'group': 'unknown',
 'ident': 'A',
 'family': 'AddInstall',
 'platform': 'Win32',
 'language': 'unknown'}

All labels are guessed using CRFSUITE conditional random fields.
For example, we would have two family labels in the example above:
"AddInstall" and "InstalleRex".

The following strategies are used to pick among the labeled antivirus names:
    - Family is guessed using TFIDF ratios for families across all documents.

    - Group and Identity are guessed by most commonly occurring groups and
      identities within AVs that guessed the picked family or guessed close to
      a picked family.  This is because the labels for group and identity only
      make sense within the confines of a specific family.

    - Platform is guessed using heuristics.

    - language, compiler, and _type are those that occur most often in the
      labeled set.

"""

import itertools
import Levenshtein
import logging
import numpy as np
import re
import scipy
import ujson as json

from collections import Counter
from collections import defaultdict
from pycrfsuite import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer

CRF_MODEL_PATH = "all_train.model"
TANIMOTO_MAPPING_PATH = "tanimoto_mapping.json"
VOCAB_PATH = "tfidf_vocab.json"
IDF_WEIGHTS_PATH = "tfidf_idf.json"

REGEX_NONWORD = re.compile("\W")
REGEX_NONWORD_SAVED = re.compile("(\W)")
UNKNOWN = 'unknown'
FAMILY = 'family'
PLATFORM = 'platform'
GROUP = 'group'
IDENT = 'ident'
GENERIC_FAMILIES = set(['Generic', 'Gen', 'GENERIC', 'Genetic'])
OTHER_GUESSABLE_TAGS = ['language', 'compiler', '_type']

AVS = set(['TotalDefense',
           'ViRobot',
           'TheHacker',
           'ClamAV',
           'CAT-QuickHeal',
           'Antiy-AVL',
           'Baidu-International',
           'Agnitum',
           'Bkav',
           'nProtect',
           'Jiangmin',
           'Commtouch',
           'F-Prot',
           'Microsoft',
           'SUPERAntiSpyware',
           'Ad-Aware',
           'Symantec',
           'AhnLab-V3',
           'Rising',
           'NANO-Antivirus',
           'Norman',
           'Ikarus',
           'Kingsoft',
           'K7AntiVirus',
           'Panda',
           'VBA32',
           'Emsisoft',
           'Fortinet',
           'F-Secure',
           'Malwarebytes',
           'MicroWorld-eScan',
           'BitDefender',
           'Avast',
           'Kaspersky',
           'DrWeb',
           'Sophos',
           'Comodo',
           'GData',
           'ESET-NOD32',
           'AVG',
           'AntiVir',
           'VIPRE',
           'McAfee'])

TEMPLATES = (
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),
    (('w',  1), ),
    (('w',  2), ),
    (('w', -1), ('w',  0)),
    (('w',  0), ('w',  1)),
    (('pos', -2), ),
    (('pos', -1), ),
    (('pos',  0), ),
    (('pos',  1), ),
    (('pos',  2), ),
    (('pos', -2), ('pos', -1)),
    (('pos', -1), ('pos',  0)),
    (('pos',  0), ('pos',  1)),
    (('pos',  1), ('pos',  2)),
    (('pos', -2), ('pos', -1), ('pos',  0)),
    (('pos', -1), ('pos',  0), ('pos',  1)),
    (('pos',  0), ('pos',  1), ('pos',  2)),
    (('av', 0), ),
    )


class NameGeneratorException(Exception):
    pass


class FamilyPostproc(object):
    def families_to_canonical(self, families):
        """Convert list of family lists to post list of postprocessed lists.
        :param list families: list of lists of families.
        :rtype: list.

        """
        all_output_families = []
        for cluster in families:
            inner_cluster = []
            for fam in cluster:
                inner_cluster.append(self.synonimous_mapping.get(fam, fam))
            all_output_families.append(inner_cluster)
        return all_output_families


class TanimotoPostproc(FamilyPostproc):
    def __init__(self, save_file=False):
        self.synonimous_mapping = json.load(open(save_file))


class EditDistancePostproc(FamilyPostproc):
    def __init__(self, similarity_func=Levenshtein.jaro_winkler, threshold=0.9):
        self.similarity_func = similarity_func
        self.threshold = threshold

    def mapping_from_one_list(self, names_list):
        """Convert list of families to list of postprocessed families.

        Uses edit distance to replace similar names with longer names
        that are < `self.threshold` edit distance away.

        :param list names_list: list of families.
        :rtype: list.

        """
        all_clusters = []
        names_list_uniq = list(set(names_list))
        indices = set(range(len(names_list_uniq)))
        while indices:
            current_idx = indices.pop()
            current_w = names_list_uniq[current_idx]
            current_cluster = [current_w]
            idxes_to_discard = set()

            for idx in indices:
                comparison_w = names_list_uniq[idx]
                if comparison_w == current_w:
                    continue

                if self.similarity_func(
                        current_w, comparison_w) > self.threshold:
                    idxes_to_discard.add(idx)
                    current_cluster.append(comparison_w)

            indices.difference_update(idxes_to_discard)
            all_clusters.append(current_cluster)
        return similar_names_from_name_clusters(all_clusters)

    def families_to_canonical(self, families):
        all_output_families = []
        for group in families:
            inner_cluster = []
            synonimous_mapping = self.mapping_from_one_list(group)
            for fam in group:
                inner_cluster.append(synonimous_mapping.get(fam, fam))
            all_output_families.append(inner_cluster)
        return all_output_families


def make_postprocessors(tani_sf=None):
    """Postprocessor factory.
    :param str tani_sf: path to saved tanimoto JSON.
    :rtype: list

    """
    save_file = tani_sf if tani_sf is not None else TANIMOTO_MAPPING_PATH
    return [EditDistancePostproc(), TanimotoPostproc(save_file=save_file)]


def _extract_features(X):
    """Extracts feature using `TEMPLATES` from a sequence of features `X`."""
    all_features = []
    for i, _ in enumerate(X):
        el_features = []
        for template in TEMPLATES:
            features_i = []
            name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
            for field, offset in template:
                p = i + offset
                if p < 0 or p >= len(X):
                    features_i = []
                    break
                features_i.append(X[p][field])
            if features_i:
                el_features.append('%s=%s' % (name, '|'.join(features_i)))
        all_features.append(el_features)

    all_features[0].append('__BOS__')
    all_features[-1].append('__EOS__')

    return all_features


def _extract_tags(tags_dict, tag):
    return [t[0] for t in tags_dict[tag] if t]


def similar_names_from_name_clusters(name_clusters):
    """Maps similar features to their best replacement.

    Takes a sequence of lists of similar strings and creats a mapping of
    all strings in that list to the longest one.
    EG. [['abc','a','b'],['cde', 'cd']] => {'a': 'abc', 'b': 'abc', 'cd': 'cde'}

    :param iterable name_clusters: iterable of lists or tuples of strings.
    :rtype: dict

    """
    d = {}
    for cluster in name_clusters:
        longest = max(cluster, key=lambda x: len(x))
        for name in cluster:
            if name != longest:
                d[name] = longest
    return d


def preprocess_av_result(av_result, av):
    """Split an av result into a list of maps for word, pos, and av if present.

    EG. take something like 'win32.malware.group' and convert to
        [{'av': 'someav', 'w': 'win32', 'pos': '0'},
         {'av': 'someav', 'w': '.', 'pos': '.'},
         {'av': 'someav', 'w': 'malware', 'pos': '1'},
         {'av': 'someav', 'w': '.', 'pos': '.'},
         {'av': 'someav', 'w': 'group', 'pos': '2'}]

    :param str av_result: string that an ativirus returns.
    :param str av: name of the AV.
    :rtype: list

    """
    split_delim = [el if el != ' ' else '_' for el in
                   REGEX_NONWORD_SAVED.split(av_result)]
    split_no_delim = REGEX_NONWORD.split(av_result)
    delims = set(split_delim) - set(split_no_delim)

    counter = 0
    tags = []
    for el in split_delim:
        if el in delims:
            tags.append(el)
        else:
            tags.append(str(counter))
            counter += 1

    if av is not None:
        return [{'w': i, 'pos': j, 'av': k} for i, j, k in
                zip(split_delim, tags, itertools.repeat(av)) if i != '']
    else:
        return [{'w': i, 'pos': j} for i, j in
                zip(split_delim, tags) if i != '']


def load_tagger(model_path):
    """Loads tagger from a CRFSUITE binary model file.

    :param str model_path: path to the binary model file.

    """
    tagger = Tagger()
    tagger.open(model_path)
    return tagger


def load_tfidf(vocab_path, idf_weights_path):
    """Loads tfidf vectorizer from its components.
    :param str vocab_path: path to the vectorizer vocabulary JSON.
    :param str idf_weights_path: path to idf weights JSON.
    :rtype: sklearn.feature_extraction.text.TfidfVectorizer

    """
    tfidf = TfidfVectorizer(analyzer=lambda x: x,
                            vocabulary=json.load(open(vocab_path)))
    idf_vector = np.array(json.load(open(idf_weights_path)))
    tfidf._tfidf._idf_diag = scipy.sparse.diags([idf_vector], [0])
    tfidf.vocabulary_ = tfidf.vocabulary
    return tfidf


def get_all_tags(av_dict, tagger):
    """Creates a dictionary of tag types to list of tags, av tuples.
    Example:
        {'SomeAntivirus': "Win32/Trojan"} =>
            {"family": [("unknown", "SomeAntivirus")],
             "platform": [("Win32", "SomeAntivirus")],
             "_type": [("Trojan", "SomeAntivirus")]
             }
    :param dict av_dict: AV dictionary to tag.
    :param pycrfsuite._pycrfsuite.Tagger tagger: tagger to use.
    :rtype: dict

    """
    all_results = defaultdict(list)

    for tag in OTHER_GUESSABLE_TAGS:
        all_results[tag] = []
    for tag in (PLATFORM, FAMILY, GROUP, IDENT):
        all_results[tag] = []

    for k, v in av_dict.items():

        if k not in AVS or v is None:
            continue

        preproc_res = preprocess_av_result(v, k)
        av_tags = tagger.tag(_extract_features(preproc_res))

        for res_dict, av_tag in zip(preproc_res, av_tags):
            all_results[av_tag].append((res_dict['w'], k))

    return dict(all_results)


def get_tag(av_dict, tagger, tag):
    """Create a list of a items tagged as `tag` in the dictionary value.
    E.G. get_tag({'SomeAntivirus': "Win32/Trojan"}, tagger, 'platform')
        => [('Win32', 'SomeAntivirus'), ]
    :param dict av_dict: AV dictionary to tag.
    :param pycrfsuite._pycrfsuite.Tagger tagger: tagger to use.
    :param str tag: tag to use.
    :rtype: list

    """
    all_results = []
    for k, v in av_dict.items():

        if k not in AVS or v is None:
            continue

        preproc_res = preprocess_av_result(v, k)
        av_tags = tagger.tag(_extract_features(preproc_res))

        for res_dict, av_tag in zip(preproc_res, av_tags):
            if av_tag == tag:
                all_results.append((res_dict['w'], k))

    return all_results


def guess_platform(av_dict, tagger, platform_tags=None):
    """Uses heuristics to guess platform from an av dictionary using a tagger.

    :param dict av_dict: AV dictionary to tag.
    :param pycrfsuite._pycrfsuite.Tagger tagger: tagger to use.
    :param list|tuple|None platform_tags: all platform tags.
    :rtype: str

    """
    WINDOWS = "Win32"
    ANDROID = "Android"

    def _decide_platform(platform_list):
        def map_to_canonical(platform_str):
            lower_str = platform_str.lower()[:3]
            if lower_str == 'win' or lower_str == 'w32' or lower_str == 'pe':
                return WINDOWS
            elif lower_str == 'and':
                return ANDROID
            else:
                return UNKNOWN

        platform_strings = [WINDOWS, ANDROID, UNKNOWN]
        p2c = {p: 0 for p in platform_strings}

        for platform in platform_list:
            p2c[map_to_canonical(platform)] += 1

        res = sorted(p2c.items(), key=lambda x: x[1], reverse=True)

        if res[0][1] == 0:
            return UNKNOWN

        for k, v in res:
            if k == UNKNOWN:
                continue
            return k

        return UNKNOWN

    if platform_tags is None:
        all_results = _extract_tags(get_tag(av_dict, tagger, PLATFORM),
                                    PLATFORM)
    else:
        all_results = platform_tags

    return _decide_platform(all_results)


def guess_by_commonality(av_dict, tagger, tag, precalculated_tags=None):
    """Guess an output tag from an av_dict based on how often it appears.

    :param dict av_dict: AV dictionary to tag.
    :param pycrfsuite._pycrfsuite.Tagger tagger: tagger to use.
    :param str tag: tag to use.
    :param list|tuple|None precalculated_tags: all precalculated tags.
    :rtype: str

    """
    tags_to_count = _extract_tags(get_tag(av_dict, tagger, tag), tag) \
        if precalculated_tags is None else precalculated_tags
    result = Counter(tags_to_count).most_common(1)
    if result:
        return result[0][0]
    else:
        return UNKNOWN


def guess_family(tfidf, tagger, av_dict, idx_to_words, postprocessors=[],
                 family_tags=None):
    """Guess family probabilities from an av_dict.

    E.G. When av_dict is
    {"F-Prot": "W32/AddInstall.A", "Comodo": "Application.Win32.InstalleRex.KG"}

    the output tuple is
    ({'AddInstall': 0.82868242670257763, 'InstalleRex': 0.55971906852842446},
     {'AddInstall': 'F-Prot', 'InstalleRex': 'Comodo'})

    :param sklearn.feature_extraction.text.TfidfVectorizer tfidf: vectorizer.
    :param pycrfsuite._pycrfsuite.Tagger tagger: tagger to use.
    :param dict av_dict: AV dictionary to tag.
    :param dict idx_to_words: index to tokens reverse dictionary.
    :param list postprocessors: list of postprocessors to use.
    :param list|tuple|None family_tags: precalculated family tags.
    :rtype: tuple

    """
    if family_tags is None:
        tags = (get_tag(av_dict, tagger, FAMILY),)
    else:
        tags = (family_tags,)

    #  get AVS because we'll need them later for word to av mapping since the
    #  order will stay the same
    avs = [t[1] for t in tags[0]]
    tags = ([t[0] for t in tags[0]], )  # get rid of AV information in tags.

    for postprocessor in postprocessors:
        tags = postprocessor.families_to_canonical(tags)

    m = tfidf.transform(tags)
    words_to_vals = {idx_to_words[idx]: val for idx, val in
                     zip(m.indices, m[0, m.indices].toarray()[0])}

    # scale Generic family heuristic
    words_to_vals.update(
        {k: v/len(words_to_vals) for k, v in words_to_vals.items()
         if k in GENERIC_FAMILIES})

    words_to_avs = defaultdict(list)
    for tag, av in zip(tags[0], avs):
        words_to_avs[tag].append(av)
    return words_to_vals, words_to_avs


def _guess_everything(tfidf, tagger, av_dict, idx_to_words, postprocessors=[]):
    """Guess all tags from an av dict.

    Eg. given av_dict
    {"F-Prot": "W32/AddInstall.A",
     "Comodo": "Application.Win32.InstalleRex.KG"}

    It would guess the following tags.
    {'group': 'unknown',
     'platform': 'Win32',
     'ident': 'A',
     'language': 'unknown',
     'family': 'AddInstall',
     '_type': 'Application',
     'compiler': 'unknown'}

    The actual tags would depend on what data the models have been trained on.

    :param sklearn.feature_extraction.text.TfidfVectorizer tfidf: vectorizer.
    :param pycrfsuite._pycrfsuite.Tagger tagger: tagger to use.
    :param dict av_dict: AV dictionary to tag.
    :param dict idx_to_words: index to tokens reverse dictionary of
                              tfidf.vocabulary_.
    :param list postprocessors: list of postprocessors to use.
    :rtype: dict

    """
    all_tags = get_all_tags(av_dict, tagger)
    family_probs, words_to_avs = guess_family(
        tfidf, tagger, av_dict, idx_to_words, postprocessors=postprocessors,
        family_tags=all_tags[FAMILY])
    families_probs_sorted = sorted(family_probs.items(), key=lambda x: x[1])

    if families_probs_sorted:
        family = families_probs_sorted.pop()[0]
        family_avs = set(words_to_avs[family])
    else:
        family = UNKNOWN
        family_avs = set()

    platform = guess_platform(av_dict, tagger,
                              platform_tags=_extract_tags(all_tags, PLATFORM))

    out_dict = {FAMILY: family, PLATFORM: platform}

    for tag in (IDENT, GROUP):  # guess group and identity only within family
        precalculated_tags = [t[0] for t in all_tags[tag]
                              if t and t[1] in family_avs]
        out_dict[tag] = guess_by_commonality(
            av_dict, tagger, tag, precalculated_tags=precalculated_tags)

    for tag in OTHER_GUESSABLE_TAGS:
        out_dict[tag] = guess_by_commonality(
            av_dict, tagger, tag, precalculated_tags=_extract_tags(all_tags,
                                                                   tag))

    return out_dict


class Guesser(object):
    """Convenience class to automatically load trained data and guess tags."""
    def __init__(self, tfidf=None, tagger=None, postprocessors=[]):
        self.tfidf = tfidf if tfidf is not None else load_tfidf(
            VOCAB_PATH, IDF_WEIGHTS_PATH)
        if type(self.tfidf) != TfidfVectorizer:
            raise NameGeneratorException("TfidfVectorizer not loaded correctly")

        self.tagger = tagger if tagger is not None else load_tagger(
            CRF_MODEL_PATH)
        if type(self.tagger) != Tagger:
            raise NameGeneratorException("Tagger not loaded correctly.")

        self.idx_to_words = {v: k for k, v in self.tfidf.vocabulary_.items()}
        try:
            self.postprocessors = make_postprocessors() if not postprocessors\
                else postprocessors
        except Exception as err:
            logging.exception(err)
            raise NameGeneratorException(err)

    def guess_everything(self, av_dict):
        """Guess all tags from an av dict.

        Eg. given av_dict
        {"F-Prot": "W32/AddInstall.A",
         "Comodo": "Application.Win32.InstalleRex.KG"}

        It would guess the following tags.
        {'group': 'unknown',
         'platform': 'Win32',
         'ident': 'A',
         'language': 'unknown',
         'family': 'AddInstall',
         '_type': 'Application',
         'compiler': 'unknown'}

        The actual tags would depend on what data the models have
        been trained on.

        :param dict av_dict: AV dictionary to tag.
        :rtype: dict

        """
        try:
            return _guess_everything(self.tfidf, self.tagger, av_dict,
                                     self.idx_to_words, self.postprocessors)
        except Exception as err:
            logging.exception(err)
            raise NameGeneratorException(err)

    def guess_family(self, av_dict):
        """Guess family probabilities from an av_dict.

        E.G. When av_dict is
        {"F-Prot": "W32/AddInstall.A",
         "Comodo": "Application.Win32.InstalleRex.KG"}

        the output dict is
        {'AddInstall': 0.82868242670257763, 'InstalleRex': 0.55971906852842446}

        :param dict av_dict: AV dictionary to tag.
        :rtype: dict

        """
        try:
            return guess_family(self.tfidf, self.tagger, av_dict,
                                self.idx_to_words, self.postprocessors)
        except Exception as err:
            logging.exception(err)
            raise NameGeneratorException(err)

