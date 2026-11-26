#include "keel/audio.h"

#include "keel/math.h"

namespace keel::audio {

namespace {

struct ScaleScan {
    f64 max_abs = 0.0;
    bool has_nan = false;
    bool has_inf = false;
};

[[nodiscard]] ScaleScan scan_abs(ArrayRef<f64> v, usize m) noexcept {
    ScaleScan res{};
    for (usize i = 0; i < m; ++i) {
        const f64 x = v[i];
        if (keel::is_nan(x)) {
            res.has_nan = true;
            return res;
        }
        const f64 ax = keel::abs(x);
        if (keel::is_inf(ax)) {
            res.has_inf = true;
        }
        if (ax > res.max_abs) {
            res.max_abs = ax;
        }
    }
    return res;
}

} // namespace

f64 spectral_centroid(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m == 0) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    const f64 inv_s = ss.has_inf ? 1.0 : (1.0 / ss.max_abs);
    f64 sum_x = 0.0;
    f64 sum_fx = 0.0;
    for (usize i = 0; i < m; ++i) {
        if (keel::is_nan(freqs[i])) {
            return kNaN;
        }
        const f64 w = spectrum[i] * inv_s;
        sum_x += w;
        sum_fx += freqs[i] * w;
    }
    return (sum_x > 0.0) ? (sum_fx / sum_x) : 0.0;
}

f64 spectral_spread(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m == 0) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    const f64 inv_s = ss.has_inf ? 1.0 : (1.0 / ss.max_abs);
    f64 sum_x = 0.0;
    f64 sum_fx = 0.0;
    for (usize i = 0; i < m; ++i) {
        if (keel::is_nan(freqs[i])) {
            return kNaN;
        }
        const f64 w = spectrum[i] * inv_s;
        sum_x += w;
        sum_fx += freqs[i] * w;
    }
    if (!(sum_x > 0.0)) {
        return keel::is_nan(sum_x) ? kNaN : 0.0;
    }
    const f64 c = sum_fx / sum_x;
    if (!keel::is_finite(c)) {
        return kNaN;
    }

    f64 max_df = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 adf = keel::abs(freqs[i] - c);
        if (adf > max_df) {
            max_df = adf;
        }
    }
    if (max_df == 0.0) {
        return 0.0;
    }
    if (!keel::is_finite(max_df)) {
        return keel::is_nan(max_df) ? kNaN : kInf;
    }

    const f64 inv_df = 1.0 / max_df;
    f64 sum_dx = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 z = (freqs[i] - c) * inv_df;
        sum_dx += (z * z) * (spectrum[i] * inv_s);
    }
    const f64 var = sum_dx / sum_x;
    if (!(var > 0.0)) {
        return keel::is_nan(var) ? kNaN : 0.0;
    }
    return max_df * keel::sqrt(var);
}

f64 spectral_skewness(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m < 2) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    const f64 inv_s = ss.has_inf ? 1.0 : (1.0 / ss.max_abs);
    f64 sum_x = 0.0;
    f64 sum_fx = 0.0;
    for (usize i = 0; i < m; ++i) {
        if (keel::is_nan(freqs[i])) {
            return kNaN;
        }
        const f64 w = spectrum[i] * inv_s;
        sum_x += w;
        sum_fx += freqs[i] * w;
    }
    if (!(sum_x > 0.0)) {
        return keel::is_nan(sum_x) ? kNaN : 0.0;
    }
    const f64 centroid = sum_fx / sum_x;
    if (!keel::is_finite(centroid)) {
        return kNaN;
    }

    f64 max_df = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 adf = keel::abs(freqs[i] - centroid);
        if (adf > max_df) {
            max_df = adf;
        }
    }
    if (!(max_df > 0.0) || !keel::is_finite(max_df)) {
        return keel::is_nan(max_df) ? kNaN : 0.0;
    }

    const f64 inv_df = 1.0 / max_df;
    f64 m2 = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 z = (freqs[i] - centroid) * inv_df;
        m2 += (z * z) * (spectrum[i] * inv_s);
    }
    const f64 spread2_scaled = m2 / sum_x;
    if (!(spread2_scaled > 0.0)) {
        return keel::is_nan(spread2_scaled) ? kNaN : 0.0;
    }
    const f64 inv_spread = inv_df / keel::sqrt(spread2_scaled);

    f64 m3 = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 u = (freqs[i] - centroid) * inv_spread;
        m3 += (u * u * u) * (spectrum[i] * inv_s);
    }
    return m3 / sum_x;
}

f64 spectral_kurtosis(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m < 2) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    const f64 inv_s = ss.has_inf ? 1.0 : (1.0 / ss.max_abs);
    f64 sum_x = 0.0;
    f64 sum_fx = 0.0;
    for (usize i = 0; i < m; ++i) {
        if (keel::is_nan(freqs[i])) {
            return kNaN;
        }
        const f64 w = spectrum[i] * inv_s;
        sum_x += w;
        sum_fx += freqs[i] * w;
    }
    if (!(sum_x > 0.0)) {
        return keel::is_nan(sum_x) ? kNaN : 0.0;
    }
    const f64 centroid = sum_fx / sum_x;
    if (!keel::is_finite(centroid)) {
        return kNaN;
    }

    f64 max_df = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 adf = keel::abs(freqs[i] - centroid);
        if (adf > max_df) {
            max_df = adf;
        }
    }
    if (!(max_df > 0.0) || !keel::is_finite(max_df)) {
        return keel::is_nan(max_df) ? kNaN : 0.0;
    }

    const f64 inv_df = 1.0 / max_df;
    f64 m2 = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 z = (freqs[i] - centroid) * inv_df;
        m2 += (z * z) * (spectrum[i] * inv_s);
    }
    const f64 spread2_scaled = m2 / sum_x;
    if (!(spread2_scaled > 0.0)) {
        return keel::is_nan(spread2_scaled) ? kNaN : 0.0;
    }
    const f64 inv_spread = inv_df / keel::sqrt(spread2_scaled);

    f64 m4 = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 u = (freqs[i] - centroid) * inv_spread;
        const f64 u2 = u * u;
        m4 += (u2 * u2) * (spectrum[i] * inv_s);
    }
    return m4 / sum_x;
}

f64 spectral_rolloff_point(ArrayRef<f64> spectrum, ArrayRef<f64> freqs, f64 percentile) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m == 0) {
        return 0.0;
    }
    if (keel::is_nan(percentile)) {
        return kNaN;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    const f64 inv_s = ss.has_inf ? 1.0 : (1.0 / ss.max_abs);
    f64 total = 0.0;
    for (usize i = 0; i < m; ++i) {
        total += spectrum[i] * inv_s;
    }
    if (!(total > 0.0)) {
        return keel::is_nan(total) ? kNaN : 0.0;
    }
    const f64 thresh = percentile * total;
    f64 accum = 0.0;
    for (usize i = 0; i < m; ++i) {
        accum += spectrum[i] * inv_s;
        if (accum >= thresh) {
            return freqs[i];
        }
    }
    return freqs[m - 1];
}

f64 spectral_decrease(ArrayRef<f64> spectrum) noexcept {
    const usize m = spectrum.size();
    if (m < 2) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    const f64 inv_s = ss.has_inf ? 1.0 : (1.0 / ss.max_abs);
    const f64 x0 = spectrum[0] * inv_s;
    f64 num = 0.0;
    f64 den = 0.0;
    for (usize k = 1; k < m; ++k) {
        const f64 xk = spectrum[k] * inv_s;
        num += (xk - x0) / as<f64>(k);
        den += xk;
    }
    return (den > 0.0) ? (num / den) : 0.0;
}

f64 spectral_slope(ArrayRef<f64> spectrum, ArrayRef<f64> freqs) noexcept {
    const usize m = keel::min(spectrum.size(), freqs.size());
    if (m < 2) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    const ScaleScan fs = scan_abs(freqs, m);
    if (ss.has_nan || fs.has_nan) {
        return kNaN;
    }
    if (ss.has_inf || fs.has_inf) {
        return kNaN;
    }
    if (ss.max_abs == 0.0 || fs.max_abs == 0.0) {
        return 0.0;
    }

    const f64 inv_s = 1.0 / ss.max_abs;
    const f64 inv_f = 1.0 / fs.max_abs;
    f64 mean_f = 0.0;
    f64 mean_x = 0.0;
    for (usize i = 0; i < m; ++i) {
        mean_f += freqs[i] * inv_f;
        mean_x += spectrum[i] * inv_s;
    }
    const f64 mf = as<f64>(m);
    mean_f /= mf;
    mean_x /= mf;

    f64 num = 0.0;
    f64 den = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 df = freqs[i] * inv_f - mean_f;
        const f64 dx = spectrum[i] * inv_s - mean_x;
        num += df * dx;
        den += df * df;
    }
    if (!(den > 0.0)) {
        return 0.0;
    }
    return (num / den) * (ss.max_abs / fs.max_abs);
}

f64 spectral_crest(ArrayRef<f64> spectrum) noexcept {
    const usize m = spectrum.size();
    if (m == 0) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    const f64 inv_s = ss.has_inf ? 1.0 : (1.0 / ss.max_abs);
    f64 mx = spectrum[0] * inv_s;
    f64 sm = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 v = spectrum[i] * inv_s;
        if (v > mx) {
            mx = v;
        }
        sm += v;
    }
    const f64 mean = sm / as<f64>(m);
    return (mean > 0.0) ? (mx / mean) : 0.0;
}

f64 spectral_entropy(ArrayRef<f64> spectrum) noexcept {
    const usize m = spectrum.size();
    if (m < 2) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    const f64 inv_s = ss.has_inf ? 1.0 : (1.0 / ss.max_abs);
    f64 sm = 0.0;
    for (usize i = 0; i < m; ++i) {
        sm += spectrum[i] * inv_s;
    }
    if (!(sm > 0.0)) {
        return keel::is_nan(sm) ? kNaN : 0.0;
    }
    f64 h = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 q = (spectrum[i] * inv_s) / sm;
        if (q > 0.0) {
            h -= q * keel::log(q);
        }
    }
    return h / keel::log(as<f64>(m));
}

f64 spectral_flatness(ArrayRef<f64> spectrum) noexcept {
    const usize m = spectrum.size();
    if (m == 0) {
        return 0.0;
    }
    const ScaleScan ss = scan_abs(spectrum, m);
    if (ss.has_nan) {
        return kNaN;
    }
    if (ss.max_abs == 0.0) {
        return 0.0;
    }
    constexpr f64 kEps = Limits<f64>::kEpsilon;
    const f64 mf = as<f64>(m);
    if (!ss.has_inf && ss.max_abs > 1e150) {
        const f64 inv_s = 1.0 / ss.max_abs;
        const f64 eps_scaled = kEps * inv_s;
        f64 sum_log = 0.0;
        f64 sum_x = 0.0;
        for (usize i = 0; i < m; ++i) {
            const f64 v = spectrum[i] * inv_s;
            sum_log += keel::log(v + eps_scaled);
            sum_x += v;
        }
        const f64 geom = keel::exp(sum_log / mf);
        const f64 arith = sum_x / mf;
        return (arith > 0.0) ? (geom / arith) : 0.0;
    }

    f64 sum_log = 0.0;
    f64 sum_x = 0.0;
    for (usize i = 0; i < m; ++i) {
        sum_log += keel::log(spectrum[i] + kEps);
        sum_x += spectrum[i];
    }
    const f64 geom = keel::exp(sum_log / mf);
    const f64 arith = sum_x / mf;
    return (arith > 0.0) ? (geom / arith) : 0.0;
}

f64 spectral_flux(ArrayRef<f64> current, ArrayRef<f64> previous, f64 norm_p) noexcept {
    const usize m = keel::min(current.size(), previous.size());
    if (m == 0 || !(norm_p > 0.0)) {
        return 0.0;
    }
    f64 max_d = 0.0;
    bool has_inf = false;
    for (usize i = 0; i < m; ++i) {
        const f64 d = keel::abs(current[i] - previous[i]);
        if (keel::is_nan(d)) {
            return kNaN;
        }
        if (keel::is_inf(d)) {
            has_inf = true;
        }
        if (d > max_d) {
            max_d = d;
        }
    }
    if (has_inf) {
        return kInf;
    }
    if (max_d == 0.0) {
        return 0.0;
    }
    if (keel::is_inf(norm_p)) {
        return max_d;
    }

    const f64 inv_d = 1.0 / max_d;
    f64 s = 0.0;
    if (norm_p == 1.0) {
        for (usize i = 0; i < m; ++i) {
            s += keel::abs(current[i] - previous[i]) * inv_d;
        }
        return max_d * s;
    }
    if (norm_p == 2.0) {
        for (usize i = 0; i < m; ++i) {
            const f64 r = keel::abs(current[i] - previous[i]) * inv_d;
            s += r * r;
        }
        return max_d * keel::sqrt(s);
    }
    for (usize i = 0; i < m; ++i) {
        const f64 r = keel::abs(current[i] - previous[i]) * inv_d;
        s += keel::pow(r, norm_p);
    }
    return max_d * keel::pow(s, 1.0 / norm_p);
}

f64 spectral_flux_c64(ArrayRef<c64> current, ArrayRef<c64> previous, f64 norm_p) noexcept {
    const usize m = keel::min(current.size(), previous.size());
    if (m == 0 || !(norm_p > 0.0)) {
        return 0.0;
    }
    f64 max_d = 0.0;
    bool has_inf = false;
    for (usize i = 0; i < m; ++i) {
        const c64 diff = current[i] - previous[i];
        if (keel::is_nan(diff.real()) || keel::is_nan(diff.imag())) {
            return kNaN;
        }
        const f64 d = keel::abs(diff);
        if (keel::is_inf(d)) {
            has_inf = true;
        }
        if (d > max_d) {
            max_d = d;
        }
    }
    if (has_inf) {
        return kInf;
    }
    if (max_d == 0.0) {
        return 0.0;
    }
    if (keel::is_inf(norm_p)) {
        return max_d;
    }

    const f64 inv_d = 1.0 / max_d;
    f64 s = 0.0;
    if (norm_p == 1.0) {
        for (usize i = 0; i < m; ++i) {
            s += keel::abs(current[i] - previous[i]) * inv_d;
        }
        return max_d * s;
    }
    if (norm_p == 2.0) {
        for (usize i = 0; i < m; ++i) {
            const f64 r = keel::abs(current[i] - previous[i]) * inv_d;
            s += r * r;
        }
        return max_d * keel::sqrt(s);
    }
    for (usize i = 0; i < m; ++i) {
        const f64 r = keel::abs(current[i] - previous[i]) * inv_d;
        s += keel::pow(r, norm_p);
    }
    return max_d * keel::pow(s, 1.0 / norm_p);
}

} // namespace keel::audio
