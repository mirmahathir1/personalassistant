# Speaking/Listening Translation Pipeline

Run:

```bash
python3 test/speaking-listening-translate-en-bn/test_pipeline.py
```

The test starts with `আমি একটা ছেলে।`, synthesizes Bangla audio, transcribes it
with `listening-bn`, translates to English, synthesizes English audio,
transcribes it with `listening-en`, translates back to Bangla, and passes only
when the final Bangla string exactly matches the input.
