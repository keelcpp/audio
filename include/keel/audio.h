#pragma once

#include "keel/complex.h"
#include "keel/core.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::audio {

enum class AudioStatus : u8 {
    kOk = 0,
    kInvalidArgument,
};

// ============================================================================
// Perceptual Frequency & Loudness Scales
// ============================================================================

[[nodiscard]] f64 hz_to_mel(f64 hz) noexcept;
[[nodiscard]] f64 mel_to_hz(f64 mel) noexcept;

[[nodiscard]] f64 hz_to_bark(f64 hz) noexcept;
[[nodiscard]] f64 bark_to_hz(f64 bark) noexcept;

[[nodiscard]] f64 hz_to_erb(f64 hz) noexcept;
[[nodiscard]] f64 erb_to_hz(f64 erb) noexcept;

// ISO 532-1 (default, iso532_2 == false) and ISO 532-2 (Table 5 PCHIP + bisection).
[[nodiscard]] f64 phon_to_sone(f64 phon, bool iso532_2 = false, Allocator* alloc = nullptr);
[[nodiscard]] f64 sone_to_phon(f64 sone, bool iso532_2 = false, Allocator* alloc = nullptr);

// Batch conversions (`out.size() >= in.size()`).
[[nodiscard]] AudioStatus hz_to_mel(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept;
[[nodiscard]] AudioStatus mel_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept;
[[nodiscard]] AudioStatus hz_to_bark(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept;
[[nodiscard]] AudioStatus bark_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept;
[[nodiscard]] AudioStatus hz_to_erb(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept;
[[nodiscard]] AudioStatus erb_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept;
[[nodiscard]] AudioStatus phon_to_sone(ArrayRef<f64> in, MutArrayRef<f64> out,
                                       bool iso532_2 = false, Allocator* alloc = nullptr);
[[nodiscard]] AudioStatus sone_to_phon(ArrayRef<f64> in, MutArrayRef<f64> out,
                                       bool iso532_2 = false, Allocator* alloc = nullptr);

// ============================================================================
// Spectral Shape Descriptors (Single Spectrum Frame)
// ============================================================================

[[nodiscard]] f64 spectral_centroid(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept;
[[nodiscard]] f64 spectral_spread(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept;
[[nodiscard]] f64 spectral_skewness(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept;
[[nodiscard]] f64 spectral_kurtosis(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept;
[[nodiscard]] f64 spectral_rolloff_point(ArrayRef<f64> spectrum, ArrayRef<f64> freqs,
                                         f64 percentile = 0.95) noexcept;
[[nodiscard]] f64 spectral_decrease(ArrayRef<f64> spectrum) noexcept;
[[nodiscard]] f64 spectral_slope(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept;
[[nodiscard]] f64 spectral_crest(ArrayRef<f64> spectrum) noexcept;
[[nodiscard]] f64 spectral_entropy(ArrayRef<f64> spectrum) noexcept;
[[nodiscard]] f64 spectral_flatness(ArrayRef<f64> spectrum) noexcept;
[[nodiscard]] f64 spectral_flux(ArrayRef<f64> current, ArrayRef<f64> previous,
                                f64 norm_p = 2.0) noexcept;
[[nodiscard]] f64 spectral_flux_c64(ArrayRef<c64> current, ArrayRef<c64> previous,
                                    f64 norm_p = 2.0) noexcept;

// ============================================================================
// Mel Filterbank & Regression Delta
// ============================================================================

// Periodic Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / N).
void hamming_periodic(MutArrayRef<f64> out) noexcept;

// Bandwidth-normalized triangular Mel filterbank (Davis & Mermelstein, O'Shaughnessy Mel).
// Writes `num_bands x (fft_len/2 + 1)` column-major weights into `fb_col_major`
// and `num_bands` band center frequencies (Hz) into `center_freqs_hz`.
[[nodiscard]] AudioStatus mel_filterbank(f64 sample_rate, usize num_bands, usize fft_len,
                                         MutArrayRef<f64> fb_col_major,
                                         MutArrayRef<f64> center_freqs_hz) noexcept;

// Furui causal regression delta filter along a single 1D channel `x`.
// Requires `window_length >= 3` and odd, `y.size() >= x.size()`.
// Optional initial state `zi` (length >= window_length - 1, or empty for zero state)
// and optional final state `zf` (length >= window_length - 1, or empty).
[[nodiscard]] AudioStatus audio_delta(ArrayRef<f64> x, usize window_length, MutArrayRef<f64> y,
                                      ArrayRef<f64> zi = {}, MutArrayRef<f64> zf = {},
                                      Allocator* alloc = nullptr);

// ============================================================================
// Framing, Cepstral Coefficients, Gammatone Filterbank, MFCC & GTCC
// ============================================================================

enum class CepstralRectification : u8 {
    kLog = 0,
    kCubicRoot,
    kNone,
};

// Number of overlapping frames for a signal of `num_samples` at `sample_rate`,
// using window length `round(sample_rate * win_sec)` and overlap `round(sample_rate * overlap_sec)`.
[[nodiscard]] usize audio_frame_count(usize num_samples, f64 sample_rate,
                                      f64 win_sec, f64 overlap_sec) noexcept;

// Slaney Mel band edges (typically 42 entries: first 13 linear, remainder log-spaced).
void slaney_mel_band_edges(MutArrayRef<f64> edges) noexcept;

// Bandwidth-normalized triangular Slaney Mel filterbank on the one-sided FFT grid `H = fft_len / 2 + 1`.
// Requires `edges.size() >= 3` and `fb_col_major.size() >= H * (edges.size() - 2)` (column-major `H x num_bands`).
[[nodiscard]] AudioStatus slaney_mel_filterbank(f64 sample_rate, usize fft_len,
                                                ArrayRef<f64> edges,
                                                MutArrayRef<f64> fb_col_major) noexcept;

// Patterson-Holdsworth 4th-order gammatone filterbank magnitude response (Slaney 1993)
// on the one-sided FFT grid `H = fft_len / 2 + 1`.
// Writes `center_freqs_hz.size() x H` row-major magnitudes into `fb_row_major`.
// When `bandwidth_normalize == true`, scales each band `i` by `2 / BW[i]`.
[[nodiscard]] AudioStatus gammatone_filterbank(f64 sample_rate,
                                               ArrayRef<f64> center_freqs_hz,
                                               usize fft_len,
                                               MutArrayRef<f64> fb_row_major,
                                               bool bandwidth_normalize = true) noexcept;

// Unitary DCT-II cepstral coefficients of a spectrogram matrix.
// `spectra_col_major` is `num_bands x num_frames` column-major (`num_bands` contiguous per frame).
// Writes `num_frames x num_coeffs` column-major coefficients into `coeffs_col_major`.
[[nodiscard]] AudioStatus cepstral_coefficients(ArrayRef<f64> spectra_col_major,
                                                usize num_bands, usize num_frames,
                                                usize num_coeffs,
                                                MutArrayRef<f64> coeffs_col_major,
                                                CepstralRectification rect = CepstralRectification::kLog,
                                                Allocator* alloc = nullptr);

[[nodiscard]] AudioStatus cepstral_coefficients_f32(ArrayRef<f32> spectra_col_major,
                                                    usize num_bands, usize num_frames,
                                                    usize num_coeffs,
                                                    MutArrayRef<f32> coeffs_col_major,
                                                    CepstralRectification rect = CepstralRectification::kLog,
                                                    Allocator* alloc = nullptr);

// Standard MFCC pipeline (30 ms periodic Hamming window, 20 ms overlap, 40-band Slaney Mel filterbank).
// Let `num_frames = audio_frame_count(x.size(), sample_rate, 0.03, 0.02)`.
// Writes `num_frames x (num_coeffs + 1)` column-major `[logE, c1..cN]` into `coeffs_col_major`,
// and optionally 9-point Furui `delta_col_major` and `delta_delta_col_major` (same shape).
[[nodiscard]] AudioStatus mfcc(ArrayRef<f64> x, f64 sample_rate, usize num_coeffs,
                               MutArrayRef<f64> coeffs_col_major,
                               MutArrayRef<f64> delta_col_major = {},
                               MutArrayRef<f64> delta_delta_col_major = {},
                               Allocator* alloc = nullptr);

// Gammatone cepstral coefficients (30 ms periodic Hamming window, 20 ms overlap, ERB gammatone bank).
// Writes `num_frames x (num_coeffs + 1)` column-major `[logE, c1..cN]` into `coeffs_col_major`,
// and optionally 9-point Furui `delta_col_major` and `delta_delta_col_major` (same shape).
[[nodiscard]] AudioStatus gtcc(ArrayRef<f64> x, f64 sample_rate, usize num_coeffs,
                               MutArrayRef<f64> coeffs_col_major,
                               MutArrayRef<f64> delta_col_major = {},
                               MutArrayRef<f64> delta_delta_col_major = {},
                               Allocator* alloc = nullptr);

// ============================================================================
// Pitch & Harmonic Ratio Estimation
// ============================================================================

enum class PitchMethod : u8 {
    kNcf = 0, // Normalized Cross-Correlation Function
    kCep,     // Real Cepstrum (Noll 1967)
    kPef,     // Pitch Estimation Filter (Gonzalez & Brookes 2011)
    kLhs,     // Log Harmonic Summation (Hermes 1988)
    kSrh,     // Summation of Residual Harmonics (Drugman & Alwan 2011)
};

// Fundamental frequency f0 estimation (52 ms window, 42 ms overlap).
// Requires `f0_out.size() >= audio_frame_count(x.size(), sample_rate, 0.052, 0.042)`.
[[nodiscard]] AudioStatus pitch(ArrayRef<f64> x, f64 sample_rate,
                                MutArrayRef<f64> f0_out,
                                f64 min_freq = 50.0, f64 max_freq = 400.0,
                                PitchMethod method = PitchMethod::kNcf,
                                Allocator* alloc = nullptr);

// MPEG-7 / Peeters harmonic ratio per frame (30 ms periodic Hamming window, 20 ms overlap).
// Requires `hr_out.size() >= audio_frame_count(x.size(), sample_rate, 0.03, 0.02)`.
[[nodiscard]] AudioStatus harmonic_ratio(ArrayRef<f64> x, f64 sample_rate,
                                         MutArrayRef<f64> hr_out,
                                         Allocator* alloc = nullptr);

} // namespace keel::audio
