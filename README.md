# keel::audio

Zero-STL C++20 audio DSP & acoustic feature extraction kernels built on [`keel`](https://github.com/keelcpp/keel).

> Note: Audio file codecs (WAV, FLAC, MP3, AIFF, AU, MIDI) live in [`keelcpp/audiofile`](https://github.com/keelcpp/audiofile).

## Features

- **Perceptual scales**: Mel (O'Shaughnessy), Bark (Traunmüller), ERB (Glasberg & Moore), Phon/Sone loudness (ISO 532-1 & ISO 532-2).
- **Spectral shape descriptors**: centroid, spread, skewness, kurtosis, rolloff point, decrease, slope, crest, entropy, flatness, and spectral flux (real & complex).
- **Audio features**: periodic Hamming window, bandwidth-normalized triangular Mel filterbank, and Furui regression `audio_delta` with streaming `zi`/`zf` state.

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
