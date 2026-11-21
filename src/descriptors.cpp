#include "keel/audio.h"

#include "keel/math.h"

namespace keel::audio {

f64 spectral_centroid(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    f64 sum_x = 0.0;
    f64 sum_fx = 0.0;
    for (usize i = 0; i < m; ++i) {
        sum_x += spectrum[i];
        sum_fx += freqs[i] * spectrum[i];
    }
    return (sum_x > 0.0) ? (sum_fx / sum_x) : 0.0;
}

f64 spectral_spread(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m == 0) return 0.0;
    const f64 c = spectral_centroid(spectrum, freqs);
    f64 sum_x = 0.0;
    f64 sum_dx = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 df = freqs[i] - c;
        sum_x += spectrum[i];
        sum_dx += df * df * spectrum[i];
    }
    return (sum_x > 0.0) ? keel::sqrt(sum_dx / sum_x) : 0.0;
}

f64 spectral_skewness(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m < 2) return 0.0;
    f64 sum_x = 0.0;
    f64 sum_fx = 0.0;
    for (usize i = 0; i < m; ++i) {
        sum_x += spectrum[i];
        sum_fx += freqs[i] * spectrum[i];
    }
    if (sum_x <= 0.0) return 0.0;
    const f64 centroid = sum_fx / sum_x;
    f64 m2 = 0.0;
    f64 m3 = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 df = freqs[i] - centroid;
        m2 += df * df * spectrum[i];
        m3 += df * df * df * spectrum[i];
    }
    const f64 spread2 = m2 / sum_x;
    if (spread2 <= 0.0) return 0.0;
    const f64 spread3 = spread2 * keel::sqrt(spread2);
    return m3 / (spread3 * sum_x);
}

f64 spectral_kurtosis(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m < 2) return 0.0;
    f64 sum_x = 0.0;
    f64 sum_fx = 0.0;
    for (usize i = 0; i < m; ++i) {
        sum_x += spectrum[i];
        sum_fx += freqs[i] * spectrum[i];
    }
    if (sum_x <= 0.0) return 0.0;
    const f64 centroid = sum_fx / sum_x;
    f64 m2 = 0.0;
    f64 m4 = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 df = freqs[i] - centroid;
        const f64 df2 = df * df;
        m2 += df2 * spectrum[i];
        m4 += df2 * df2 * spectrum[i];
    }
    const f64 spread2 = m2 / sum_x;
    if (spread2 <= 0.0) return 0.0;
    const f64 spread4 = spread2 * spread2;
    return m4 / (spread4 * sum_x);
}

f64 spectral_rolloff_point(ArrayRef<f64> spectrum, ArrayRef<f64> freqs, f64 percentile) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m == 0) return 0.0;
    f64 total = 0.0;
    for (usize i = 0; i < m; ++i) {
        total += spectrum[i];
    }
    if (total <= 0.0) return 0.0;
    const f64 thresh = percentile * total;
    f64 accum = 0.0;
    for (usize i = 0; i < m; ++i) {
        accum += spectrum[i];
        if (accum >= thresh) return freqs[i];
    }
    return freqs[m - 1];
}

f64 spectral_decrease(ArrayRef<f64> spectrum) noexcept {
    const usize m = spectrum.size();
    if (m < 2) return 0.0;
    const f64 x0 = spectrum[0];
    f64 num = 0.0;
    f64 den = 0.0;
    for (usize k = 1; k < m; ++k) {
        num += (spectrum[k] - x0) / static_cast<f64>(k);
        den += spectrum[k];
    }
    return (den > 0.0) ? (num / den) : 0.0;
}

f64 spectral_slope(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m < 2) return 0.0;
    f64 mean_f = 0.0;
    f64 mean_x = 0.0;
    for (usize i = 0; i < m; ++i) {
        mean_f += freqs[i];
        mean_x += spectrum[i];
    }
    const f64 mf = static_cast<f64>(m);
    mean_f /= mf;
    mean_x /= mf;
    f64 num = 0.0;
    f64 den = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 df = freqs[i] - mean_f;
        num += df * (spectrum[i] - mean_x);
        den += df * df;
    }
    return (den > 0.0) ? (num / den) : 0.0;
}

f64 spectral_crest(ArrayRef<f64> spectrum) noexcept {
    const usize m = spectrum.size();
    if (m == 0) return 0.0;
    f64 mx = spectrum[0];
    f64 sm = 0.0;
    for (usize i = 0; i < m; ++i) {
        if (spectrum[i] > mx) mx = spectrum[i];
        sm += spectrum[i];
    }
    const f64 mean = sm / static_cast<f64>(m);
    return (mean > 0.0) ? (mx / mean) : 0.0;
}

f64 spectral_entropy(ArrayRef<f64> spectrum) noexcept {
    const usize m = spectrum.size();
    if (m < 2) return 0.0;
    f64 sm = 0.0;
    for (usize i = 0; i < m; ++i) {
        sm += spectrum[i];
    }
    if (sm <= 0.0) return 0.0;
    f64 h = 0.0;
    const f64 inv_log2 = 1.0 / kLn2;
    for (usize i = 0; i < m; ++i) {
        const f64 q = spectrum[i] / sm;
        if (q > 0.0) {
            h -= q * keel::log(q) * inv_log2;
        }
    }
    return h / (keel::log(static_cast<f64>(m)) * inv_log2);
}

f64 spectral_flatness(ArrayRef<f64> spectrum) noexcept {
    const usize m = spectrum.size();
    if (m == 0) return 0.0;
    constexpr f64 eps = Limits<f64>::kEpsilon;
    f64 sum_log = 0.0;
    f64 sum_x = 0.0;
    for (usize i = 0; i < m; ++i) {
        sum_log += keel::log(spectrum[i] + eps);
        sum_x += spectrum[i];
    }
    const f64 mf = static_cast<f64>(m);
    const f64 geom = keel::exp(sum_log / mf);
    const f64 arith = sum_x / mf;
    return (arith > 0.0) ? (geom / arith) : 0.0;
}

f64 spectral_flux(ArrayRef<f64> current, ArrayRef<f64> previous, f64 norm_p) noexcept {
    const usize m = keel::min(current.size(), previous.size());
    if (m == 0 || !(norm_p > 0.0)) return 0.0;
    f64 s = 0.0;
    for (usize i = 0; i < m; ++i) {
        s += keel::pow(keel::abs(current[i] - previous[i]), norm_p);
    }
    return keel::pow(s, 1.0 / norm_p);
}

f64 spectral_flux_c64(ArrayRef<c64> current, ArrayRef<c64> previous, f64 norm_p) noexcept {
    const usize m = keel::min(current.size(), previous.size());
    if (m == 0 || !(norm_p > 0.0)) return 0.0;
    f64 s = 0.0;
    for (usize i = 0; i < m; ++i) {
        s += keel::pow(keel::abs(current[i] - previous[i]), norm_p);
    }
    return keel::pow(s, 1.0 / norm_p);
}

} // namespace keel::audio
