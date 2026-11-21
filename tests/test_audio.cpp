#include "keel/audio.h"

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

} // namespace

int main() {
    test_scales();
    test_descriptors();
    test_features();

    if (g_failures == 0) {
        std::printf("ALL keel::audio TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
