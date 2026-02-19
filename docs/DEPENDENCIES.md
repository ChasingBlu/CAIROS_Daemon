# Dependencies

## Python (Track A)
- Python 3.10+ (tested)
- numpy 1.26.4
- torch 2.3.1
- transformers 4.41.2
- scipy (optional, for Welch t-test p-values)

**Optional GPU stack (CUDA builds):**
- torchaudio 2.7.1+cu128
- torchvision 0.22.1+cu128

## C/C++ (Track B)
- CMake 3.20+
- C++17 compiler
  - MSVC 19.3+ (Windows)
  - Clang 14+ (Linux/macOS)
  - GCC 11+ (Linux)
- ONNX Runtime 1.24.1 (C++ API)
- SentencePiece 0.1.99+ (C++ library)

## CUDA Toolchain (Optional, GPU)
- CUDA Toolkit 12.8/12.9
- `nvcc` 12.x (tested in internal runs with 12.9.41)

## OpenCV (Optional)
- OpenCV 4.x (only required for non-core visualization tooling; not used by Track A/B core pipelines)

## Toolchain Notes
- Define `ONNXRUNTIME_DIR` and `SENTENCEPIECE_DIR` at configure time.
- SecureLogger v2.0 is **not** included (private).
