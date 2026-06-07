# Bangla API Smoke Test

Run:

```bash
python3 test/api-bn/test_api_bn.py
```

The test translates a hardcoded Bangla arithmetic question to English, asks the
local proxy API, translates the API response back to Bangla, and passes when the
final Bangla response contains `২৭৯৯`.
