# virus-names
Virus names generator figures out the best family, type, language, platform, etc for a virus when given names by different AV vendors.

Loads trained models and guesses virus names.

Overall, the idea is to guess a name for a virus from the names that Antiviruses
give it.

When using the Guesser class, all Exceptions are wrapped in a
`NameGeneratorException`.

Usage:
```python
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
```

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

- `language`, `compiler`, and `_type` are those that occur most often in the
  labeled set.
