# English Speaking/Listening Round Trip

Run:

```bash
python3 test/speaking-listening-en/test_roundtrip.py
```

The script synthesizes the hardcoded English sentence in `test_roundtrip.py`
with `speaking-en`, writes `generated-en.wav`, transcribes the in-memory audio
with `listening-en`, and prints exact plus normalized match results.
