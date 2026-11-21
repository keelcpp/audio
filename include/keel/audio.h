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

} // namespace keel::audio
