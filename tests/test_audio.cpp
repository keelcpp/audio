#include "keel/audio.h"

#include "keel/array.h"
#include "keel/format.h"
#include "keel/fs.h"
#include "keel/math.h"
#include "keel/string.h"

namespace {

using namespace keel;
namespace aud = keel::audio;

i32 g_failures = 0;

void check_true(bool cond, const char* expr, i32 line) {
    if (!cond) {
        keel::eprintln("FAIL line %d: %s", line, expr);
        ++g_failures;
    }
}

void check_near(f64 actual, f64 expected, f64 tol, const char* expr, i32 line) {
    if (!(keel::abs(actual - expected) <= tol)) {
        keel::eprintln("FAIL line %d: %s (actual=%.15g, expected=%.15g, tol=%.15g)", line, expr, actual, expected, tol);
        ++g_failures;
    }
}

#define CHECK(cond) check_true((cond), #cond, __LINE__)
#define CHECK_NEAR(a, e, t) check_near((a), (e), (t), #a " ~= " #e, __LINE__)

template <typename T>
bool write_bin(StringRef dir, StringRef name, const T* data, usize count) {
    PathBuf path;
    keel::path_join_into(path, dir, name);
    auto file_res = File::open(StringRef(path.data(), path.size()), FileMode::Write);
    if (!file_res.is_ok()) {
        keel::eprintln("Failed to open output file: %s", path.c_str());
        return false;
    }
    File f = keel::move(file_res.value());
    if (count > 0) {
        const auto* bytes = reinterpret_cast<const u8*>(data);
        f.write(ArrayRef<u8>(bytes, count * sizeof(T)));
    }
    const Error err = f.close();
    if (!err.is_ok()) {
        keel::eprintln("Failed to close output file: %s", path.c_str());
        return false;
    }
    return true;
}

void test_scales() {
    // Mel / ERB roundtrip including tiny near-zero values (log1p/expm1 accuracy)
    const f64 freqs[6] = {0.0, 1e-9, 440.0, 1000.0, 8000.0, 20000.0};
    for (usize i = 0; i < 6; ++i) {
        CHECK_NEAR(aud::mel_to_hz(aud::hz_to_mel(freqs[i])), freqs[i], 1e-11);
        CHECK_NEAR(aud::erb_to_hz(aud::hz_to_erb(freqs[i])), freqs[i], 1e-11);
    }
    CHECK_NEAR(aud::hz_to_mel(1000.0), 2595.0 * keel::log10(1.0 + 1000.0 / 700.0), 1e-12);

    // Bark monotonicity, range, roundtrip, and huge/Inf stability
    CHECK(aud::hz_to_bark(100.0) < aud::hz_to_bark(1000.0));
    CHECK(aud::hz_to_bark(1000.0) < aud::hz_to_bark(10000.0));
    CHECK(aud::bark_to_hz(10.0) > 1000.0);
    for (usize i = 0; i < 6; ++i) {
        CHECK_NEAR(aud::bark_to_hz(aud::hz_to_bark(freqs[i])), freqs[i], 1e-9);
    }
    CHECK_NEAR(aud::hz_to_bark(kInf), 1.22 * 26.28 - 4.422, 1e-12);
    CHECK(keel::is_finite(aud::hz_to_bark(1e200)));

    // NaN propagation across all scales
    CHECK(keel::is_nan(aud::hz_to_mel(kNaN)));
    CHECK(keel::is_nan(aud::mel_to_hz(kNaN)));
    CHECK(keel::is_nan(aud::hz_to_bark(kNaN)));
    CHECK(keel::is_nan(aud::bark_to_hz(kNaN)));
    CHECK(keel::is_nan(aud::hz_to_erb(kNaN)));
    CHECK(keel::is_nan(aud::erb_to_hz(kNaN)));
    CHECK(keel::is_nan(aud::phon_to_sone(kNaN, false)));
    CHECK(keel::is_nan(aud::phon_to_sone(kNaN, true)));
    CHECK(keel::is_nan(aud::sone_to_phon(kNaN, false)));
    CHECK(keel::is_nan(aud::sone_to_phon(kNaN, true)));
    CHECK(keel::is_nan(aud::phon_to_sone(-1.0, false)));
    CHECK(keel::is_nan(aud::phon_to_sone(-1.0, true)));
    CHECK(keel::is_nan(aud::sone_to_phon(-1.0, false)));
    CHECK(keel::is_nan(aud::sone_to_phon(-1.0, true)));

    // ISO 532-1 loudness roundtrip
    CHECK_NEAR(aud::phon_to_sone(0.0, false), 0.0, 1e-14);
    CHECK_NEAR(aud::sone_to_phon(0.0, false), 0.0, 1e-14);
    CHECK_NEAR(aud::phon_to_sone(40.0, false), 1.0, 1e-14);
    CHECK_NEAR(aud::sone_to_phon(1.0, false), 40.0, 1e-14);
    CHECK_NEAR(aud::phon_to_sone(60.0, false), 4.0, 1e-14);
    CHECK_NEAR(aud::sone_to_phon(aud::phon_to_sone(25.0, false), false), 25.0, 1e-12);
    CHECK_NEAR(aud::sone_to_phon(aud::phon_to_sone(75.0, false), false), 75.0, 1e-12);

    // ISO 532-2 loudness roundtrip including p=0 and p > 120 / p > 144 (regression: uncapped linear extrap)
    const f64 phons[8] = {0.0, 1.0, 20.0, 40.0, 75.0, 110.0, 130.0, 150.0};
    f64 sones[8]{};
    f64 back[8]{};
    CHECK(aud::phon_to_sone(phons, sones, true) == aud::AudioStatus::kOk);
    CHECK(aud::sone_to_phon(sones, back, true) == aud::AudioStatus::kOk);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(back[i], phons[i], 1e-9);
    }
    CHECK_NEAR(sones[0], 0.001, 1e-15);
    CHECK_NEAR(sones[3], 1.0, 1e-14);
    CHECK_NEAR(sones[6], 562.8, 1e-11);
    CHECK_NEAR(sones[7], 1013.2, 1e-11);
    CHECK_NEAR(aud::sone_to_phon(0.0, true), 0.0, 1e-15);
    CHECK_NEAR(aud::sone_to_phon(0.0005, true), 0.0, 1e-15);

    // Overlapping slice aliasing safety in batch conversions
    f64 buf[6] = {100.0, 500.0, 1000.0, 2000.0, 4000.0, 0.0};
    f64 ref_mel[5]{};
    CHECK(aud::hz_to_mel(ArrayRef<f64>(buf, 5), MutArrayRef<f64>(ref_mel, 5)) == aud::AudioStatus::kOk);
    CHECK(aud::hz_to_mel(ArrayRef<f64>(buf, 5), MutArrayRef<f64>(buf + 1, 5)) == aud::AudioStatus::kOk);
    for (usize i = 0; i < 5; ++i) {
        CHECK_NEAR(buf[i + 1], ref_mel[i], 1e-14);
    }
    // Buffer too small rejected
    CHECK(aud::hz_to_mel(ArrayRef<f64>(buf, 5), MutArrayRef<f64>(ref_mel, 4)) == aud::AudioStatus::kInvalidArgument);
}

void test_descriptors() {
    const f64 freqs[5] = {0.0, 100.0, 200.0, 300.0, 400.0};
    // Symmetric impulse at 200 Hz
    const f64 tone[5] = {0.0, 0.0, 1.0, 0.0, 0.0};
    CHECK_NEAR(aud::spectral_centroid(tone, freqs), 200.0, 1e-14);
    CHECK_NEAR(aud::spectral_spread(tone, freqs), 0.0, 1e-14);
    CHECK_NEAR(aud::spectral_rolloff_point(tone, freqs, 0.95), 200.0, 1e-14);
    CHECK_NEAR(aud::spectral_crest(tone), 5.0, 1e-14);
    CHECK_NEAR(aud::spectral_entropy(tone), 0.0, 1e-14);

    // Flat spectrum
    const f64 flat[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    CHECK_NEAR(aud::spectral_centroid(flat, freqs), 200.0, 1e-14);
    CHECK_NEAR(aud::spectral_skewness(flat, freqs), 0.0, 1e-14);
    CHECK_NEAR(aud::spectral_slope(flat, freqs), 0.0, 1e-14);
    CHECK_NEAR(aud::spectral_decrease(flat), 0.0, 1e-14);
    CHECK_NEAR(aud::spectral_crest(flat), 1.0, 1e-14);
    CHECK_NEAR(aud::spectral_entropy(flat), 1.0, 1e-14);
    CHECK_NEAR(aud::spectral_flatness(flat), 1.0, 1e-12);
    CHECK(aud::spectral_kurtosis(flat, freqs) > 1.0);

    // Overflow / underflow scaling immunity (1e250 spectrum & 1e100 freqs)
    const f64 huge_s[5] = {0.2e250, 1.5e250, 3.0e250, 0.8e250, 0.5e250};
    const f64 unit_s[5] = {0.2, 1.5, 3.0, 0.8, 0.5};
    const f64 huge_f[5] = {0.0, 1e100, 2e100, 3e100, 4e100};
    CHECK_NEAR(aud::spectral_centroid(huge_s, freqs), aud::spectral_centroid(unit_s, freqs), 1e-12);
    CHECK_NEAR(aud::spectral_spread(huge_s, freqs), aud::spectral_spread(unit_s, freqs), 1e-12);
    CHECK_NEAR(aud::spectral_skewness(huge_s, huge_f), aud::spectral_skewness(unit_s, freqs), 1e-12);
    CHECK_NEAR(aud::spectral_kurtosis(huge_s, huge_f), aud::spectral_kurtosis(unit_s, freqs), 1e-12);
    CHECK_NEAR(aud::spectral_decrease(huge_s), aud::spectral_decrease(unit_s), 1e-12);
    CHECK_NEAR(aud::spectral_crest(huge_s), aud::spectral_crest(unit_s), 1e-12);
    CHECK_NEAR(aud::spectral_entropy(huge_s), aud::spectral_entropy(unit_s), 1e-12);
    CHECK_NEAR(aud::spectral_flatness(huge_s), aud::spectral_flatness(unit_s), 1e-12);
    CHECK_NEAR(aud::spectral_rolloff_point(huge_s, freqs, 0.95), aud::spectral_rolloff_point(unit_s, freqs, 0.95), 1e-12);

    // NaN propagation
    const f64 nan_s[5] = {1.0, kNaN, 1.0, 1.0, 1.0};
    CHECK(keel::is_nan(aud::spectral_centroid(nan_s, freqs)));
    CHECK(keel::is_nan(aud::spectral_spread(nan_s, freqs)));
    CHECK(keel::is_nan(aud::spectral_skewness(nan_s, freqs)));
    CHECK(keel::is_nan(aud::spectral_kurtosis(nan_s, freqs)));
    CHECK(keel::is_nan(aud::spectral_rolloff_point(nan_s, freqs, 0.95)));
    CHECK(keel::is_nan(aud::spectral_rolloff_point(unit_s, freqs, kNaN)));
    CHECK(keel::is_nan(aud::spectral_decrease(nan_s)));
    CHECK(keel::is_nan(aud::spectral_slope(nan_s, freqs)));
    CHECK(keel::is_nan(aud::spectral_crest(nan_s)));
    CHECK(keel::is_nan(aud::spectral_entropy(nan_s)));
    CHECK(keel::is_nan(aud::spectral_flatness(nan_s)));

    // Flux (L1, L2, L_inf, overflow-safe, complex, NaN)
    const f64 s1[3] = {1.0, 2.0, 3.0};
    const f64 s2[3] = {4.0, 6.0, 3.0};
    CHECK_NEAR(aud::spectral_flux(s2, s1, 2.0), 5.0, 1e-14);
    CHECK_NEAR(aud::spectral_flux(s2, s1, 1.0), 7.0, 1e-14);
    CHECK_NEAR(aud::spectral_flux(s2, s1, kInf), 4.0, 1e-14);
    const f64 big1[2] = {0.0, 0.0};
    const f64 big2[2] = {3e200, 4e200};
    CHECK_NEAR(aud::spectral_flux(big2, big1, 2.0), 5e200, 1e188);
    CHECK(keel::is_nan(aud::spectral_flux(nan_s, flat, 2.0)));

    const c64 c1[2] = {c64(1.0, 2.0), c64(-3.0, 4.0)};
    const c64 c2[2] = {c64(4.0, 6.0), c64(-3.0, 4.0)};
    CHECK_NEAR(aud::spectral_flux_c64(c2, c1, 2.0), 5.0, 1e-14);
    CHECK_NEAR(aud::spectral_flux_c64(c2, c1, kInf), 5.0, 1e-14);

    // Empty and zero inputs
    const f64 zeros[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK_NEAR(aud::spectral_centroid(zeros, freqs), 0.0, 1e-15);
    CHECK_NEAR(aud::spectral_spread(zeros, freqs), 0.0, 1e-15);
    CHECK_NEAR(aud::spectral_flatness(zeros), 0.0, 1e-15);
    CHECK_NEAR(aud::spectral_centroid({}, {}), 0.0, 1e-15);
}

void test_features() {
    // Periodic Hamming including N=1 regression (MATLAB hamming(1,'periodic') == 1)
    f64 w1[1]{};
    aud::hamming_periodic(w1);
    CHECK_NEAR(w1[0], 1.0, 1e-15);

    f64 w[4]{};
    aud::hamming_periodic(w);
    CHECK_NEAR(w[0], 0.08, 1e-14);
    CHECK_NEAR(w[1], 0.54, 1e-14);
    CHECK_NEAR(w[2], 1.00, 1e-14);
    CHECK_NEAR(w[3], 0.54, 1e-14);

    // Mel filterbank
    constexpr usize B = 8;
    constexpr usize NFFT = 64;
    constexpr usize H = NFFT / 2 + 1;
    f64 fb[B * H]{};
    f64 cf[B]{};
    CHECK(aud::mel_filterbank(16000.0, B, NFFT, fb, cf) == aud::AudioStatus::kOk);
    for (usize b = 0; b < B; ++b) {
        CHECK(cf[b] > 0.0 && cf[b] < 8000.0);
        if (b > 0) CHECK(cf[b] > cf[b - 1]);
        f64 row_sum = 0.0;
        for (usize k = 0; k < H; ++k) {
            CHECK(fb[b + k * B] >= 0.0);
            row_sum += fb[b + k * B];
        }
        CHECK(row_sum > 0.0);
    }
    // Overflow & invalid arg guards in mel_filterbank
    CHECK(aud::mel_filterbank(16000.0, Limits<usize>::kMax, 64, fb, cf) == aud::AudioStatus::kInvalidArgument);
    CHECK(aud::mel_filterbank(kNaN, B, NFFT, fb, cf) == aud::AudioStatus::kInvalidArgument);
    CHECK(aud::mel_filterbank(16000.0, B, 0, fb, cf) == aud::AudioStatus::kInvalidArgument);

    // audio_delta chunk continuity & aliasing (in-place and forward-overlapping)
    const f64 x[8] = {1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0};
    f64 y_full[8]{};
    CHECK(aud::audio_delta(x, 5, y_full) == aud::AudioStatus::kOk);

    f64 y_part[8]{};
    f64 zf[4]{};
    CHECK(aud::audio_delta(ArrayRef<f64>(x, 3), 5, MutArrayRef<f64>(y_part, 3), {}, zf) ==
          aud::AudioStatus::kOk);
    CHECK(aud::audio_delta(ArrayRef<f64>(x + 3, 5), 5, MutArrayRef<f64>(y_part + 3, 5), zf, zf) ==
          aud::AudioStatus::kOk);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(y_part[i], y_full[i], 1e-14);
    }

    // In-place & forward-overlapping aliasing
    f64 alias_buf[9] = {1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 0.0};
    CHECK(aud::audio_delta(ArrayRef<f64>(alias_buf, 8), 5, MutArrayRef<f64>(alias_buf + 1, 8)) ==
          aud::AudioStatus::kOk);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(alias_buf[i + 1], y_full[i], 1e-14);
    }

    // Invalid window length rejected
    CHECK(aud::audio_delta(x, 4, y_full) == aud::AudioStatus::kInvalidArgument);
    CHECK(aud::audio_delta(x, 1, y_full) == aud::AudioStatus::kInvalidArgument);
}

void test_cepstral_mfcc_gtcc() {
    // Slaney band edges
    f64 edges[42]{};
    aud::slaney_mel_band_edges(edges);
    CHECK_NEAR(edges[0], 133.33333333333333, 1e-12);
    for (usize i = 1; i < 42; ++i) {
        CHECK(edges[i] > edges[i - 1]);
    }

    // Slaney filterbank + NaN edge robustness
    constexpr usize NFFT = 480;
    constexpr usize H = NFFT / 2 + 1;
    Array<f64> fb(H * 40);
    CHECK(aud::slaney_mel_filterbank(16000.0, NFFT, edges, fb) == aud::AudioStatus::kOk);
    for (usize b = 0; b < 40; ++b) {
        f64 sum = 0.0;
        for (usize j = 0; j < H; ++j) {
            CHECK(fb[j + b * H] >= 0.0);
            sum += fb[j + b * H];
        }
        CHECK(sum > 0.0);
    }
    const f64 nan_edges[3] = {100.0, kNaN, 300.0};
    Array<f64> fb1(H);
    CHECK(aud::slaney_mel_filterbank(16000.0, NFFT, nan_edges, fb1) == aud::AudioStatus::kOk);
    for (usize j = 0; j < H; ++j) {
        CHECK_NEAR(fb1[j], 0.0, 1e-15);
    }

    // Gammatone filterbank + high-bandwidth / invalid-cf guards
    const f64 fc[4] = {200.0, 600.0, 1500.0, 3500.0};
    Array<f64> gb(4 * H);
    CHECK(aud::gammatone_filterbank(16000.0, fc, NFFT, gb, true) == aud::AudioStatus::kOk);
    for (usize b = 0; b < 4; ++b) {
        f64 max_v = 0.0;
        usize max_k = 0;
        for (usize k = 0; k < H; ++k) {
            const f64 v = gb[b * H + k];
            CHECK(keel::is_finite(v) && v >= 0.0);
            if (v > max_v) {
                max_v = v;
                max_k = k;
            }
        }
        const f64 peak_hz = keel::as<f64>(max_k) * 16000.0 / keel::as<f64>(NFFT);
        CHECK(keel::abs(peak_hz - fc[b]) < 100.0);
    }
    // Extreme b_rad * t (regression: no exp(+inf) overflow)
    const f64 fc_hi[1] = {5000.0};
    Array<f64> gb_hi(H);
    CHECK(aud::gammatone_filterbank(100.0, fc_hi, NFFT, gb_hi, true) == aud::AudioStatus::kOk);
    for (usize k = 0; k < H; ++k) {
        CHECK(keel::is_finite(gb_hi[k]) && gb_hi[k] >= 0.0);
    }
    const f64 fc_bad[1] = {-10.0};
    CHECK(aud::gammatone_filterbank(16000.0, fc_bad, NFFT, gb_hi, true) == aud::AudioStatus::kInvalidArgument);
    const f64 fc_nan[1] = {kNaN};
    CHECK(aud::gammatone_filterbank(16000.0, fc_nan, NFFT, gb_hi, true) == aud::AudioStatus::kInvalidArgument);

    // Cepstral coefficients (f64 and f32, including f32 zero-entry floor regression)
    const f64 spec[8] = {1.0, 2.0, 4.0, 8.0, 8.0, 4.0, 2.0, 1.0};
    f64 c_log[6]{};
    f64 c_cbrt[6]{};
    f64 c_none[6]{};
    CHECK(aud::cepstral_coefficients(spec, 4, 2, 3, c_log, aud::CepstralRectification::kLog) ==
          aud::AudioStatus::kOk);
    CHECK(aud::cepstral_coefficients(spec, 4, 2, 3, c_cbrt, aud::CepstralRectification::kCubicRoot) ==
          aud::AudioStatus::kOk);
    CHECK(aud::cepstral_coefficients(spec, 4, 2, 3, c_none, aud::CepstralRectification::kNone) ==
          aud::AudioStatus::kOk);
    CHECK(keel::is_finite(c_log[0]) && keel::is_finite(c_cbrt[0]) && keel::is_finite(c_none[0]));
    // Negative rejected for log/cubic-root, allowed for none
    const f64 neg_spec[4] = {1.0, -1.0, 2.0, 3.0};
    f64 c_bad[2]{};
    CHECK(aud::cepstral_coefficients(neg_spec, 4, 1, 2, c_bad, aud::CepstralRectification::kLog) ==
          aud::AudioStatus::kInvalidArgument);
    CHECK(aud::cepstral_coefficients(neg_spec, 4, 1, 2, c_bad, aud::CepstralRectification::kNone) ==
          aud::AudioStatus::kOk);
    // Overflow guard on num_bands * num_frames
    CHECK(aud::cepstral_coefficients(spec, Limits<usize>::kMax / 2, 4, 2, c_bad, aud::CepstralRectification::kLog) ==
          aud::AudioStatus::kInvalidArgument);

    const f32 spec_f[8] = {1.0f, 2.0f, 4.0f, 8.0f, 8.0f, 4.0f, 2.0f, 1.0f};
    f32 cf_out[6]{};
    CHECK(aud::cepstral_coefficients_f32(spec_f, 4, 2, 3, cf_out, aud::CepstralRectification::kLog) ==
          aud::AudioStatus::kOk);
    for (usize i = 0; i < 6; ++i) {
        CHECK_NEAR(keel::as<f64>(cf_out[i]), c_log[i], 1e-5);
    }
    // f32 zero entry uses f32 min-positive floor (~ -37.92978), matching MATLAB single()
    const f32 spec_f0[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    f32 cf0_out[3]{};
    CHECK(aud::cepstral_coefficients_f32(spec_f0, 4, 1, 3, cf0_out, aud::CepstralRectification::kLog) ==
          aud::AudioStatus::kOk);
    CHECK_NEAR(keel::as<f64>(cf0_out[0]), -18.5758152008057, 1e-4);

    // MFCC and GTCC on a 0.25s 16kHz synthetic tone
    constexpr f64 kFs = 16000.0;
    constexpr usize kN = 4000;
    Array<f64> sig(kN);
    for (usize i = 0; i < kN; ++i) {
        const f64 t = keel::as<f64>(i) / kFs;
        sig[i] = 0.6 * keel::sin(kTau * 440.0 * t) + 0.3 * keel::sin(kTau * 880.0 * t);
    }
    const usize n_frames = aud::audio_frame_count(kN, kFs, 0.03, 0.02);
    CHECK(n_frames > 10);
    constexpr usize kNC = 13;
    Array<f64> mfcc_c(n_frames * (kNC + 1));
    Array<f64> mfcc_d(n_frames * (kNC + 1));
    Array<f64> mfcc_dd(n_frames * (kNC + 1));
    CHECK(aud::mfcc(sig, kFs, kNC, mfcc_c, mfcc_d, mfcc_dd) == aud::AudioStatus::kOk);
    for (usize i = 0; i < mfcc_c.size(); ++i) {
        CHECK(keel::is_finite(mfcc_c[i]));
        CHECK(keel::is_finite(mfcc_d[i]));
        CHECK(keel::is_finite(mfcc_dd[i]));
    }
    // Atomic error: undersized delta must fail before mutating coeffs
    Array<f64> sentinel(n_frames * (kNC + 1), -999.0);
    f64 tiny_delta[1]{};
    CHECK(aud::mfcc(sig, kFs, kNC, sentinel, tiny_delta) == aud::AudioStatus::kInvalidArgument);
    CHECK_NEAR(sentinel[0], -999.0, 1e-15);

    Array<f64> gtcc_c(n_frames * (kNC + 1));
    Array<f64> gtcc_d(n_frames * (kNC + 1));
    Array<f64> gtcc_dd(n_frames * (kNC + 1));
    CHECK(aud::gtcc(sig, kFs, kNC, gtcc_c, gtcc_d, gtcc_dd) == aud::AudioStatus::kOk);
    for (usize i = 0; i < gtcc_c.size(); ++i) {
        CHECK(keel::is_finite(gtcc_c[i]));
        CHECK(keel::is_finite(gtcc_d[i]));
        CHECK(keel::is_finite(gtcc_dd[i]));
    }
    CHECK(aud::gtcc(sig, kFs, kNC, sentinel, tiny_delta) == aud::AudioStatus::kInvalidArgument);
    CHECK_NEAR(sentinel[0], -999.0, 1e-15);

    // Huge sample_rate overflow safety in audio_frame_count
    CHECK(aud::audio_frame_count(1000, 1e300, 0.03, 0.02) == 0);
}

void test_pitch_and_harmonic_ratio() {
    constexpr f64 kFs = 8000.0;
    constexpr usize kN = 1600; // 0.2 s
    constexpr f64 kF0 = 200.0;
    Array<f64> tone(kN);
    for (usize i = 0; i < kN; ++i) {
        const f64 t = keel::as<f64>(i) / kFs;
        tone[i] = 1.0 * keel::sin(kTau * kF0 * t) + 0.5 * keel::sin(kTau * 2.0 * kF0 * t) +
                  0.3 * keel::sin(kTau * 3.0 * kF0 * t) + 0.2 * keel::sin(kTau * 4.0 * kF0 * t);
    }

    const usize n_pitch = aud::audio_frame_count(kN, kFs, 0.052, 0.042);
    CHECK(n_pitch > 5);
    Array<f64> f0(n_pitch);

    const aud::PitchMethod methods[5] = {
        aud::PitchMethod::kNcf,
        aud::PitchMethod::kCep,
        aud::PitchMethod::kPef,
        aud::PitchMethod::kLhs,
        aud::PitchMethod::kSrh,
    };
    for (usize m = 0; m < 5; ++m) {
        CHECK(aud::pitch(tone, kFs, f0, 50.0, 400.0, methods[m]) == aud::AudioStatus::kOk);
        for (usize i = 1; i + 1 < n_pitch; ++i) {
            CHECK(keel::abs(f0[i] - kF0) < 10.0);
        }
    }

    // Extreme amplitude scaling immunity (1e180) for NCF, CEP, PEF, SRH, HR
    Array<f64> huge_tone(kN);
    for (usize i = 0; i < kN; ++i) {
        huge_tone[i] = tone[i] * 1e180;
    }
    Array<f64> f0_huge(n_pitch);
    CHECK(aud::pitch(huge_tone, kFs, f0_huge, 50.0, 400.0, aud::PitchMethod::kNcf) == aud::AudioStatus::kOk);
    CHECK(keel::abs(f0_huge[n_pitch / 2] - kF0) < 10.0);
    CHECK(aud::pitch(huge_tone, kFs, f0_huge, 50.0, 400.0, aud::PitchMethod::kCep) == aud::AudioStatus::kOk);
    CHECK(keel::abs(f0_huge[n_pitch / 2] - kF0) < 10.0);
    CHECK(aud::pitch(huge_tone, kFs, f0_huge, 50.0, 400.0, aud::PitchMethod::kPef) == aud::AudioStatus::kOk);
    CHECK(keel::abs(f0_huge[n_pitch / 2] - kF0) < 10.0);
    CHECK(aud::pitch(huge_tone, kFs, f0_huge, 50.0, 400.0, aud::PitchMethod::kSrh) == aud::AudioStatus::kOk);
    CHECK(keel::abs(f0_huge[n_pitch / 2] - kF0) < 10.0);

    // Constant signal (rank-deficient autocorrelation for Levinson-Durbin in SRH)
    Array<f64> dc_sig(kN, 2.5);
    CHECK(aud::pitch(dc_sig, kFs, f0_huge, 50.0, 400.0, aud::PitchMethod::kSrh) == aud::AudioStatus::kOk);
    for (usize i = 0; i < n_pitch; ++i) {
        CHECK(keel::is_finite(f0_huge[i]));
    }

    // High max_freq > fs (regression: NCF must never select lag 0; LHS/SRH must not overflow i64)
    CHECK(aud::pitch(tone, kFs, f0_huge, 50.0, 1e18, aud::PitchMethod::kNcf) == aud::AudioStatus::kOk);
    for (usize i = 0; i < n_pitch; ++i) {
        CHECK(keel::is_finite(f0_huge[i]) && f0_huge[i] > 0.0);
    }
    CHECK(aud::pitch(tone, kFs, f0_huge, 50.0, 1e18, aud::PitchMethod::kLhs) == aud::AudioStatus::kInvalidArgument);
    CHECK(aud::pitch(tone, kFs, f0_huge, 50.0, 1e18, aud::PitchMethod::kSrh) == aud::AudioStatus::kOk);

    const usize n_hr = aud::audio_frame_count(kN, kFs, 0.03, 0.02);
    CHECK(n_hr > 5);
    Array<f64> hr(n_hr);
    CHECK(aud::harmonic_ratio(tone, kFs, hr) == aud::AudioStatus::kOk);
    for (usize i = 0; i < n_hr; ++i) {
        CHECK(hr[i] > 0.75 && hr[i] <= 1.0);
    }
    Array<f64> hr_huge(n_hr);
    CHECK(aud::harmonic_ratio(huge_tone, kFs, hr_huge) == aud::AudioStatus::kOk);
    for (usize i = 0; i < n_hr; ++i) {
        CHECK(hr_huge[i] > 0.75 && hr_huge[i] <= 1.0);
    }

    // Higher frequency pure tone (440 Hz)
    Array<f64> tone440(kN);
    for (usize i = 0; i < kN; ++i) {
        tone440[i] = keel::sin(kTau * 440.0 * keel::as<f64>(i) / kFs);
    }
    CHECK(aud::harmonic_ratio(tone440, kFs, hr) == aud::AudioStatus::kOk);
    for (usize i = 0; i < n_hr; ++i) {
        CHECK(hr[i] > 0.9 && hr[i] <= 1.0);
    }
}

i32 dump_reference(StringRef dir) {
    // 1. Perceptual scales
    {
        const f64 hz[10] = {0.0, 50.0, 100.0, 440.0, 1000.0, 2500.0, 4000.0, 8000.0, 12000.0, 20000.0};
        f64 mel[10]{}, mel_rt[10]{}, bark[10]{}, bark_rt[10]{}, erb[10]{}, erb_rt[10]{};
        if (aud::hz_to_mel(hz, mel) != aud::AudioStatus::kOk) return 3;
        if (aud::mel_to_hz(mel, mel_rt) != aud::AudioStatus::kOk) return 3;
        if (aud::hz_to_bark(hz, bark) != aud::AudioStatus::kOk) return 3;
        if (aud::bark_to_hz(bark, bark_rt) != aud::AudioStatus::kOk) return 3;
        if (aud::hz_to_erb(hz, erb) != aud::AudioStatus::kOk) return 3;
        if (aud::erb_to_hz(erb, erb_rt) != aud::AudioStatus::kOk) return 3;

        const f64 phons[13] = {0.0, 1.0, 2.2, 12.0, 25.0, 40.0, 60.0, 75.0, 110.0, 120.0, 130.0, 144.0, 150.0};
        f64 s1[13]{}, s2[13]{};
        if (aud::phon_to_sone(phons, s1, false) != aud::AudioStatus::kOk) return 3;
        if (aud::phon_to_sone(phons, s2, true) != aud::AudioStatus::kOk) return 3;

        const f64 sones[12] = {0.0, 0.0005, 0.001, 0.05, 0.26, 1.0, 4.0, 10.0, 151.0, 337.6, 400.0, 1000.0};
        f64 p1[12]{}, p2[12]{};
        if (aud::sone_to_phon(sones, p1, false) != aud::AudioStatus::kOk) return 3;
        if (aud::sone_to_phon(sones, p2, true) != aud::AudioStatus::kOk) return 3;

        if (!write_bin(dir, "scale_hz.bin", hz, 10)) return 2;
        if (!write_bin(dir, "scale_mel.bin", mel, 10)) return 2;
        if (!write_bin(dir, "scale_mel_rt.bin", mel_rt, 10)) return 2;
        if (!write_bin(dir, "scale_bark.bin", bark, 10)) return 2;
        if (!write_bin(dir, "scale_bark_rt.bin", bark_rt, 10)) return 2;
        if (!write_bin(dir, "scale_erb.bin", erb, 10)) return 2;
        if (!write_bin(dir, "scale_erb_rt.bin", erb_rt, 10)) return 2;
        if (!write_bin(dir, "scale_phon.bin", phons, 13)) return 2;
        if (!write_bin(dir, "scale_sone_532_1.bin", s1, 13)) return 2;
        if (!write_bin(dir, "scale_sone_532_2.bin", s2, 13)) return 2;
        if (!write_bin(dir, "scale_sone_in.bin", sones, 12)) return 2;
        if (!write_bin(dir, "scale_phon_532_1.bin", p1, 12)) return 2;
        if (!write_bin(dir, "scale_phon_532_2.bin", p2, 12)) return 2;
    }

    // 2. Spectral descriptors
    {
        constexpr usize M = 16;
        f64 freqs[M]{}, s1[M]{}, s2[M]{};
        c64 c1[M]{}, c2[M]{};
        for (usize i = 0; i < M; ++i) {
            freqs[i] = as<f64>(i) * 125.0;
            const f64 x = as<f64>(i + 1);
            s1[i] = 0.25 + 2.0 * keel::exp(-0.15 * (x - 6.0) * (x - 6.0)) + 0.4 * keel::sin(0.7 * x) * keel::sin(0.7 * x);
            s2[i] = 0.30 + 1.7 * keel::exp(-0.12 * (x - 8.0) * (x - 8.0)) + 0.5 * keel::cos(0.5 * x) * keel::cos(0.5 * x);
            c1[i] = c64(s1[i] * keel::cos(0.3 * x), s1[i] * keel::sin(0.3 * x));
            c2[i] = c64(s2[i] * keel::cos(0.6 * x), -s2[i] * keel::sin(0.4 * x));
        }
        const f64 vals[13] = {
            aud::spectral_centroid(s1, freqs),
            aud::spectral_spread(s1, freqs),
            aud::spectral_skewness(s1, freqs),
            aud::spectral_kurtosis(s1, freqs),
            aud::spectral_rolloff_point(s1, freqs, 0.95),
            aud::spectral_rolloff_point(s1, freqs, 0.50),
            aud::spectral_decrease(s1),
            aud::spectral_slope(s1, freqs),
            aud::spectral_crest(s1),
            aud::spectral_entropy(s1),
            aud::spectral_flatness(s1),
            aud::spectral_flux(s2, s1, 2.0),
            aud::spectral_flux(s2, s1, 1.0),
        };
        const f64 extra[4] = {
            aud::spectral_flux(s2, s1, kInf),
            aud::spectral_flux_c64(c2, c1, 2.0),
            aud::spectral_flux_c64(c2, c1, 1.0),
            aud::spectral_flux_c64(c2, c1, kInf),
        };
        if (!write_bin(dir, "desc_freqs.bin", freqs, M)) return 2;
        if (!write_bin(dir, "desc_s1.bin", s1, M)) return 2;
        if (!write_bin(dir, "desc_s2.bin", s2, M)) return 2;
        if (!write_bin(dir, "desc_c1.bin", c1, M)) return 2;
        if (!write_bin(dir, "desc_c2.bin", c2, M)) return 2;
        if (!write_bin(dir, "desc_vals.bin", vals, 13)) return 2;
        if (!write_bin(dir, "desc_extra.bin", extra, 4)) return 2;
    }

    // 3. Features: Hamming, Mel filterbank, audio_delta
    {
        f64 ham[64]{};
        aud::hamming_periodic(ham);
        if (!write_bin(dir, "hamming_64.bin", ham, 64)) return 2;

        constexpr usize B = 12;
        constexpr usize NFFT = 128;
        constexpr usize H = NFFT / 2 + 1;
        Array<f64> mel_fb(B * H);
        Array<f64> mel_cf(B);
        if (aud::mel_filterbank(16000.0, B, NFFT, mel_fb, mel_cf) != aud::AudioStatus::kOk) return 3;
        if (!write_bin(dir, "mel_fb.bin", mel_fb.data(), mel_fb.size())) return 2;
        if (!write_bin(dir, "mel_cf.bin", mel_cf.data(), mel_cf.size())) return 2;

        f64 dx[32]{}, dy5[32]{}, dy9[32]{};
        for (usize i = 0; i < 32; ++i) {
            const f64 t = as<f64>(i);
            dx[i] = 0.1 * t * t - 0.7 * t + 2.0 * keel::sin(0.35 * t);
        }
        if (aud::audio_delta(dx, 5, dy5) != aud::AudioStatus::kOk) return 3;
        if (aud::audio_delta(dx, 9, dy9) != aud::AudioStatus::kOk) return 3;
        if (!write_bin(dir, "delta_x.bin", dx, 32)) return 2;
        if (!write_bin(dir, "delta_y5.bin", dy5, 32)) return 2;
        if (!write_bin(dir, "delta_y9.bin", dy9, 32)) return 2;
    }

    // 4. Cepstral / Slaney / Gammatone / MFCC / GTCC
    {
        f64 edges[42]{};
        aud::slaney_mel_band_edges(edges);
        constexpr usize NFFT = 480;
        constexpr usize H = NFFT / 2 + 1;
        Array<f64> sfb(H * 40);
        if (aud::slaney_mel_filterbank(16000.0, NFFT, edges, sfb) != aud::AudioStatus::kOk) return 3;
        if (!write_bin(dir, "slaney_edges.bin", edges, 42)) return 2;
        if (!write_bin(dir, "slaney_fb.bin", sfb.data(), sfb.size())) return 2;

        const f64 gt_fc[4] = {200.0, 600.0, 1500.0, 3500.0};
        Array<f64> gt_fb(4 * H), gt_fb_unnorm(4 * H);
        if (aud::gammatone_filterbank(16000.0, gt_fc, NFFT, gt_fb, true) != aud::AudioStatus::kOk) return 3;
        if (aud::gammatone_filterbank(16000.0, gt_fc, NFFT, gt_fb_unnorm, false) != aud::AudioStatus::kOk) return 3;
        if (!write_bin(dir, "gt_fc.bin", gt_fc, 4)) return 2;
        if (!write_bin(dir, "gt_fb.bin", gt_fb.data(), gt_fb.size())) return 2;
        if (!write_bin(dir, "gt_fb_unnorm.bin", gt_fb_unnorm.data(), gt_fb_unnorm.size())) return 2;

        // Cepstral coefficients (8 bands x 5 frames, 4 coeffs)
        f64 cspec[40]{};
        for (usize f = 0; f < 5; ++f) {
            for (usize b = 0; b < 8; ++b) {
                cspec[b + f * 8] = 0.5 + as<f64>(b + 1) * (1.0 + 0.3 * as<f64>(f)) +
                                   0.4 * keel::cos(0.7 * as<f64>(b) + 0.5 * as<f64>(f));
            }
        }
        f64 c_log[20]{}, c_cbrt[20]{}, c_none[20]{};
        if (aud::cepstral_coefficients(cspec, 8, 5, 4, c_log, aud::CepstralRectification::kLog) != aud::AudioStatus::kOk) return 3;
        if (aud::cepstral_coefficients(cspec, 8, 5, 4, c_cbrt, aud::CepstralRectification::kCubicRoot) != aud::AudioStatus::kOk) return 3;
        if (aud::cepstral_coefficients(cspec, 8, 5, 4, c_none, aud::CepstralRectification::kNone) != aud::AudioStatus::kOk) return 3;
        if (!write_bin(dir, "cep_spec.bin", cspec, 40)) return 2;
        if (!write_bin(dir, "cep_log.bin", c_log, 20)) return 2;
        if (!write_bin(dir, "cep_cbrt.bin", c_cbrt, 20)) return 2;
        if (!write_bin(dir, "cep_none.bin", c_none, 20)) return 2;

        const f32 cspec_f32[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 2.5f, 1.25f, 0.5f};
        f32 c_log_f32[6]{};
        if (aud::cepstral_coefficients_f32(cspec_f32, 4, 2, 3, c_log_f32, aud::CepstralRectification::kLog) !=
            aud::AudioStatus::kOk) return 3;
        if (!write_bin(dir, "cep_spec_f32.bin", cspec_f32, 8)) return 2;
        if (!write_bin(dir, "cep_log_f32.bin", c_log_f32, 6)) return 2;

        // MFCC & GTCC on 16 kHz signal
        constexpr f64 kFs = 16000.0;
        constexpr usize kN = 4000;
        Array<f64> sig(kN);
        for (usize i = 0; i < kN; ++i) {
            const f64 t = as<f64>(i) / kFs;
            sig[i] = 0.6 * keel::sin(kTau * 440.0 * t) + 0.3 * keel::sin(kTau * 880.0 * t) +
                     0.15 * keel::cos(kTau * 1320.0 * t);
        }
        const usize nf = aud::audio_frame_count(kN, kFs, 0.03, 0.02);
        constexpr usize kNC = 13;
        Array<f64> mc(nf * (kNC + 1)), md(nf * (kNC + 1)), mdd(nf * (kNC + 1));
        Array<f64> gc(nf * (kNC + 1)), gd(nf * (kNC + 1)), gdd(nf * (kNC + 1));
        if (aud::mfcc(sig, kFs, kNC, mc, md, mdd) != aud::AudioStatus::kOk) return 3;
        if (aud::gtcc(sig, kFs, kNC, gc, gd, gdd) != aud::AudioStatus::kOk) return 3;
        if (!write_bin(dir, "sig_16k.bin", sig.data(), sig.size())) return 2;
        if (!write_bin(dir, "mfcc_c.bin", mc.data(), mc.size())) return 2;
        if (!write_bin(dir, "mfcc_d.bin", md.data(), md.size())) return 2;
        if (!write_bin(dir, "mfcc_dd.bin", mdd.data(), mdd.size())) return 2;
        if (!write_bin(dir, "gtcc_c.bin", gc.data(), gc.size())) return 2;
        if (!write_bin(dir, "gtcc_d.bin", gd.data(), gd.size())) return 2;
        if (!write_bin(dir, "gtcc_dd.bin", gdd.data(), gdd.size())) return 2;
    }

    // 5. Pitch & Harmonic Ratio on 8 kHz harmonic tone
    {
        constexpr f64 kFs = 8000.0;
        constexpr usize kN = 1600;
        constexpr f64 kF0 = 200.0;
        Array<f64> tone(kN);
        for (usize i = 0; i < kN; ++i) {
            const f64 t = as<f64>(i) / kFs;
            tone[i] = 1.0 * keel::sin(kTau * kF0 * t) + 0.5 * keel::sin(kTau * 2.0 * kF0 * t) +
                      0.3 * keel::sin(kTau * 3.0 * kF0 * t) + 0.2 * keel::sin(kTau * 4.0 * kF0 * t);
        }
        const usize np = aud::audio_frame_count(kN, kFs, 0.052, 0.042);
        Array<f64> p_ncf(np), p_cep(np), p_pef(np), p_lhs(np), p_srh(np);
        if (aud::pitch(tone, kFs, p_ncf, 50.0, 400.0, aud::PitchMethod::kNcf) != aud::AudioStatus::kOk) return 3;
        if (aud::pitch(tone, kFs, p_cep, 50.0, 400.0, aud::PitchMethod::kCep) != aud::AudioStatus::kOk) return 3;
        if (aud::pitch(tone, kFs, p_pef, 50.0, 400.0, aud::PitchMethod::kPef) != aud::AudioStatus::kOk) return 3;
        if (aud::pitch(tone, kFs, p_lhs, 50.0, 400.0, aud::PitchMethod::kLhs) != aud::AudioStatus::kOk) return 3;
        if (aud::pitch(tone, kFs, p_srh, 50.0, 400.0, aud::PitchMethod::kSrh) != aud::AudioStatus::kOk) return 3;

        const usize nhr = aud::audio_frame_count(kN, kFs, 0.03, 0.02);
        Array<f64> hr(nhr);
        if (aud::harmonic_ratio(tone, kFs, hr) != aud::AudioStatus::kOk) return 3;

        if (!write_bin(dir, "sig_8k.bin", tone.data(), tone.size())) return 2;
        if (!write_bin(dir, "pitch_ncf.bin", p_ncf.data(), p_ncf.size())) return 2;
        if (!write_bin(dir, "pitch_cep.bin", p_cep.data(), p_cep.size())) return 2;
        if (!write_bin(dir, "pitch_pef.bin", p_pef.data(), p_pef.size())) return 2;
        if (!write_bin(dir, "pitch_lhs.bin", p_lhs.data(), p_lhs.size())) return 2;
        if (!write_bin(dir, "pitch_srh.bin", p_srh.data(), p_srh.size())) return 2;
        if (!write_bin(dir, "hr_8k.bin", hr.data(), hr.size())) return 2;
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && StringRef(argv[1]) == "--dump-ref") {
        return dump_reference(StringRef(argv[2]));
    }

    test_scales();
    test_descriptors();
    test_features();
    test_cepstral_mfcc_gtcc();
    test_pitch_and_harmonic_ratio();

    if (g_failures == 0) {
        keel::println("ALL keel::audio TESTS PASSED");
        return 0;
    }
    keel::eprintln("TOTAL FAILURES: %d", g_failures);
    return 1;
}
