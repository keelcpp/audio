#include "keel/audio.h"

#include "keel/array.h"
#include "keel/math.h"

#include <cstdio>

namespace {

using namespace keel;
namespace aud = keel::audio;

i32 g_failures = 0;

void check_true(bool cond, const char* expr, i32 line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failures;
    }
}

void check_near(f64 actual, f64 expected, f64 tol, const char* expr, i32 line) {
    if (!(keel::abs(actual - expected) <= tol)) {
        std::fprintf(stderr, "FAIL line %d: %s (actual=%.15g, expected=%.15g, tol=%.3g)\n",
                     line, expr, actual, expected, tol);
        ++g_failures;
    }
}

#define CHECK(cond) check_true((cond), #cond, __LINE__)
#define CHECK_NEAR(a, e, t) check_near((a), (e), (t), #a " ~= " #e, __LINE__)

void test_scales() {
    // Mel roundtrip
    const f64 freqs[4] = {0.0, 440.0, 1000.0, 8000.0};
    for (usize i = 0; i < 4; ++i) {
        CHECK_NEAR(aud::mel_to_hz(aud::hz_to_mel(freqs[i])), freqs[i], 1e-11);
        CHECK_NEAR(aud::erb_to_hz(aud::hz_to_erb(freqs[i])), freqs[i], 1e-11);
    }
    CHECK_NEAR(aud::hz_to_mel(1000.0), 2595.0 * keel::log10(1.0 + 1000.0 / 700.0), 1e-12);

    // Bark monotonicity and range
    CHECK(aud::hz_to_bark(100.0) < aud::hz_to_bark(1000.0));
    CHECK(aud::hz_to_bark(1000.0) < aud::hz_to_bark(10000.0));
    CHECK(aud::bark_to_hz(10.0) > 1000.0);

    // ISO 532-1 loudness roundtrip
    CHECK_NEAR(aud::phon_to_sone(40.0, false), 1.0, 1e-14);
    CHECK_NEAR(aud::sone_to_phon(1.0, false), 40.0, 1e-14);
    CHECK_NEAR(aud::phon_to_sone(60.0, false), 4.0, 1e-14);
    CHECK_NEAR(aud::sone_to_phon(aud::phon_to_sone(25.0, false), false), 25.0, 1e-12);
    CHECK_NEAR(aud::sone_to_phon(aud::phon_to_sone(75.0, false), false), 75.0, 1e-12);

    // ISO 532-2 loudness roundtrip
    const f64 phons[4] = {20.0, 40.0, 75.0, 110.0};
    f64 sones[4]{};
    f64 back[4]{};
    CHECK(aud::phon_to_sone(phons, sones, true) == aud::AudioStatus::kOk);
    CHECK(aud::sone_to_phon(sones, back, true) == aud::AudioStatus::kOk);
    for (usize i = 0; i < 4; ++i) {
        CHECK_NEAR(back[i], phons[i], 1e-9);
    }
    CHECK_NEAR(sones[1], 1.0, 1e-12);
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

    // Flux
    const f64 s1[3] = {1.0, 2.0, 3.0};
    const f64 s2[3] = {4.0, 6.0, 3.0};
    CHECK_NEAR(aud::spectral_flux(s2, s1, 2.0), 5.0, 1e-14);
    CHECK_NEAR(aud::spectral_flux(s2, s1, 1.0), 7.0, 1e-14);
}

void test_features() {
    // Periodic Hamming
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

    // audio_delta chunk continuity
    const f64 x[8] = {1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0};
    f64 y_full[8]{};
    CHECK(aud::audio_delta(x, 5, y_full) == aud::AudioStatus::kOk);

    f64 y_part[8]{};
    f64 zf[4]{};
    CHECK(aud::audio_delta(ArrayRef<f64>(x, 3), 5, MutArrayRef<f64>(y_part, 3), {}, zf) ==
          aud::AudioStatus::kOk);
    CHECK(aud::audio_delta(ArrayRef<f64>(x + 3, 5), 5, MutArrayRef<f64>(y_part + 3, 5), zf, {}) ==
          aud::AudioStatus::kOk);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(y_part[i], y_full[i], 1e-14);
    }

    // Invalid window length rejected
    CHECK(aud::audio_delta(x, 4, y_full) == aud::AudioStatus::kInvalidArgument);
}

void test_cepstral_mfcc_gtcc() {
    // Slaney band edges
    f64 edges[42]{};
    aud::slaney_mel_band_edges(edges);
    CHECK_NEAR(edges[0], 133.33333333333333, 1e-12);
    for (usize i = 1; i < 42; ++i) {
        CHECK(edges[i] > edges[i - 1]);
    }

    // Slaney filterbank
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

    // Gammatone filterbank
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

    // Cepstral coefficients (f64 and f32)
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

    const f32 spec_f[8] = {1.0f, 2.0f, 4.0f, 8.0f, 8.0f, 4.0f, 2.0f, 1.0f};
    f32 cf_out[6]{};
    CHECK(aud::cepstral_coefficients_f32(spec_f, 4, 2, 3, cf_out, aud::CepstralRectification::kLog) ==
          aud::AudioStatus::kOk);
    for (usize i = 0; i < 6; ++i) {
        CHECK_NEAR(keel::as<f64>(cf_out[i]), c_log[i], 1e-5);
    }

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

    Array<f64> gtcc_c(n_frames * (kNC + 1));
    Array<f64> gtcc_d(n_frames * (kNC + 1));
    Array<f64> gtcc_dd(n_frames * (kNC + 1));
    CHECK(aud::gtcc(sig, kFs, kNC, gtcc_c, gtcc_d, gtcc_dd) == aud::AudioStatus::kOk);
    for (usize i = 0; i < gtcc_c.size(); ++i) {
        CHECK(keel::is_finite(gtcc_c[i]));
        CHECK(keel::is_finite(gtcc_d[i]));
        CHECK(keel::is_finite(gtcc_dd[i]));
    }
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
        // Check middle frames match ~200 Hz within 10 Hz
        for (usize i = 1; i + 1 < n_pitch; ++i) {
            CHECK(keel::abs(f0[i] - kF0) < 10.0);
        }
    }

    const usize n_hr = aud::audio_frame_count(kN, kFs, 0.03, 0.02);
    CHECK(n_hr > 5);
    Array<f64> hr(n_hr);
    CHECK(aud::harmonic_ratio(tone, kFs, hr) == aud::AudioStatus::kOk);
    for (usize i = 0; i < n_hr; ++i) {
        CHECK(hr[i] > 0.75 && hr[i] <= 1.0);
    }

    // Higher frequency pure tone (440 Hz) has shorter period relative to 30ms Hamming window -> > 0.9
    Array<f64> tone440(kN);
    for (usize i = 0; i < kN; ++i) {
        tone440[i] = keel::sin(kTau * 440.0 * keel::as<f64>(i) / kFs);
    }
    CHECK(aud::harmonic_ratio(tone440, kFs, hr) == aud::AudioStatus::kOk);
    for (usize i = 0; i < n_hr; ++i) {
        CHECK(hr[i] > 0.9 && hr[i] <= 1.0);
    }
}

} // namespace

int main() {
    test_scales();
    test_descriptors();
    test_features();
    test_cepstral_mfcc_gtcc();
    test_pitch_and_harmonic_ratio();

    if (g_failures == 0) {
        std::printf("ALL keel::audio TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
