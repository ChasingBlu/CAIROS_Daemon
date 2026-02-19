# SecureLogger Notice (Private Protocol)

SecureLogger v2.0 is a private, non-open-source security protocol. This repository **does not** include its implementation. The Track A/B pipelines expose hooks and optional flags to integrate SecureLogger, but all secure logging is disabled by default.

If you have access to SecureLogger v2.0, place the module on your build/PYTHONPATH and enable `CAIROS_SECURE_LOGGER` (Python) or `-DCAIROS_WITH_SECURE_LOGGER=ON` (CMake).
