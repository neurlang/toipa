# toipa
To ipa transformer based project. Experimentally turn a written language to ipa.

# sk2ipa
Slovak language to ipa. This is an example of a language which is word based.

This is not really ipa as one letter is different which is easier for me to
read. I think it was ť - θ, i prefer this to the ť - c.

Furthermore there are special placeholders. They are:

A - abbrevation. Should be expanded someday.
F - foreign word, read as in the foreign language.

# jp2ipa
Japanese to ipa. This is an example for a whole sententce being fed to the
transformer language.

# installation

```
pip install -r ./requirements.txt
```

Now you can train the models using the provided `train.sh` script. If your
session is interrupted, you can resume later using the `resume.sh` script.

Once done, you can generate the dictionary for every word using the `dict.sh`
script, or do inference for whole sentences using the `eval.sh` script.
