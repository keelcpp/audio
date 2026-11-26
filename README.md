# keel::audio

Zero-STL C++20 audio DSP & acoustic feature extraction kernels built on [`keel`](https://github.com/keelcpp/keel), [`keel::interp`](https://github.com/keelcpp/interp), and [`keel::fft`](https://github.com/keelcpp/fft).

> Note: Audio file codecs (WAV, FLAC, MP3, AIFF, AU, MIDI) live in [`keelcpp/audiofile`](https://github.com/keelcpp/audiofile).

## Features

- **Perceptual scales**: Mel (O'Shaughnessy), Bark (Traunmüller), ERB (Glasberg & Moore), Phon/Sone loudness (ISO 532-1 & ISO 532-2 with exact monotonic PCHIP/linear inversion and overlap-safe batch conversions).
- **Spectral shape descriptors**: overflow-safe and NaN-aware centroid, spread, skewness, kurtosis, rolloff point, decrease, slope, crest, entropy, flatness, and spectral flux (`L1`, `L2`, `Lp`, `Linf` for real `f64` and complex `c64`).
- **Audio features**: periodic Hamming window, bandwidth-normalized triangular Mel filterbank (O'Shaughnessy & Slaney), Patterson–Holdsworth 4th-order gammatone filterbank, unitary DCT-II cepstral coefficients (`f64` & `f32`), full MFCC & GTCC pipelines, and Furui regression `audio_delta` with streaming `zi`/`zf` state.
- **Pitch & harmonics**: fundamental frequency estimation (`pitch`: NCF, CEP, PEF, LHS, SRH) and MPEG-7 / Peeters `harmonic_ratio`.

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Reference Verification

Differential verification against NumPy/SciPy and MATLAB Audio Toolbox:

```bash
python tests/verify_reference.py build/Release/test_audio.exe
```
