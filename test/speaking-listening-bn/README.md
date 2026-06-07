# Bangla Speaking/Listening Round Trip

Run:

```bash
python3 test/speaking-listening-bn/test_roundtrip.py
```

The script synthesizes the hardcoded Bangla sentence in `test_roundtrip.py`
with `speaking-bn`, writes `generated-bn.wav`, transcribes the in-memory audio
with `listening-bn`, and prints exact plus normalized match results.
