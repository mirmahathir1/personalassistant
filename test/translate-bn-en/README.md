# English/Bangla Translation Round Trip

Run:

```bash
python3 test/translate-bn-en/test_roundtrip.py
```

The test translates `I'm a boy.` with `translate-en-to-bn`, translates the
Bangla result back with `translate-bn-to-en`, and passes only when the final
English string exactly matches the original input.
