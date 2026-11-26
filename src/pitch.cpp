#include "keel/audio.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/complex.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::audio {

namespace {

struct FrameSpec {
    usize win_len = 0;
    usize overlap = 0;
    usize hop = 1;
    usize num_frames = 0;
};

[[nodiscard]] bool mul_overflows(usize a, usize b) noexcept {
    return a != 0 && b > Limits<usize>::kMax / a;
}

FrameSpec make_frame_spec(usize n, f64 fs, f64 win_sec, f64 ov_sec) noexcept {
    FrameSpec s{};
    if (!(fs > 0.0) || !keel::is_finite(fs)) {
        return s;
    }
    const f64 w_f = fs * win_sec;
    const f64 o_f = fs * ov_sec;
    constexpr f64 kMaxSafeSamples = 4503599627370496.0; // 2^52
    if (!keel::is_finite(w_f) || !keel::is_finite(o_f) || w_f >= kMaxSafeSamples || o_f >= kMaxSafeSamples) {
        return s;
    }
    const i64 w_i = keel::round_i64(w_f);
    const i64 o_i = keel::round_i64(o_f);
    if (w_i <= 0) {
        return s;
    }
    s.win_len = as<usize>(w_i);
    s.overlap = (o_i > 0) ? as<usize>(o_i) : 0;
    if (s.overlap >= s.win_len) {
        s.overlap = s.win_len - 1;
    }
    s.hop = (s.win_len > s.overlap) ? (s.win_len - s.overlap) : 1;
    s.num_frames = (n >= s.win_len) ? ((n - s.win_len) / s.hop + 1) : 0;
    return s;
}

usize next_pow2(usize x) noexcept {
    if (x <= 1) {
        return 1;
    }
    constexpr usize kMaxPow2 = usize{1} << (sizeof(usize) * 8 - 1);
    if (x > kMaxPow2) {
        return 0;
    }
    usize p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

[[nodiscard]] f64 frame_max_abs(const f64* x, usize len) noexcept {
    f64 mx = 0.0;
    for (usize i = 0; i < len; ++i) {
        const f64 ax = keel::abs(x[i]);
        if (keel::is_nan(ax)) {
            return kNaN;
        }
        if (ax > mx) {
            mx = ax;
        }
    }
    return mx;
}

void autocorr_lags(const f64* x, usize len, usize max_lag, f64* r_out) noexcept {
    for (usize k = 0; k < max_lag; ++k) {
        f64 s = 0.0;
        for (usize n = 0; n + k < len; ++n) {
            s += x[n] * x[n + k];
        }
        r_out[k] = s;
    }
}

void partial_power_lags(const f64* x, usize len, usize max_lag, f64* p_out) noexcept {
    for (usize k = 0; k < max_lag; ++k) {
        f64 s = 0.0;
        for (usize n = 0; n + k < len; ++n) {
            const f64 v = x[n + k];
            s += v * v;
        }
        p_out[k] = s;
    }
}

// Levinson-Durbin AR model from biased autocorrelation (order p).
void lpc_levinson(const f64* x, usize len, usize p, f64* a_out, Allocator* scratch_alloc) {
    for (usize i = 0; i <= p; ++i) {
        a_out[i] = 0.0;
    }
    a_out[0] = 1.0;
    if (p == 0 || len == 0) {
        return;
    }

    const f64 max_x = frame_max_abs(x, len);
    if (!(max_x > 0.0) || !keel::is_finite(max_x)) {
        return;
    }
    const f64 inv_x = 1.0 / max_x;

    Array<f64> r(p + 1, 0.0, scratch_alloc);
    const f64 inv_n = 1.0 / as<f64>(len);
    for (usize k = 0; k <= p; ++k) {
        f64 s = 0.0;
        for (usize i = 0; i + k < len; ++i) {
            s += (x[i] * inv_x) * (x[i + k] * inv_x);
        }
        r[k] = s * inv_n;
    }

    if (!(r[0] > 0.0) || !keel::is_finite(r[0])) {
        return;
    }

    Array<f64> a_prev(p + 1, 0.0, scratch_alloc);
    Array<f64> a_next(p + 1, 0.0, scratch_alloc);
    a_prev[0] = 1.0;
    f64 e = r[0];
    const f64 e_tol = Limits<f64>::kMinPositive;

    for (usize i = 1; i <= p; ++i) {
        if (!(e > e_tol) || !keel::is_finite(e)) {
            break;
        }
        f64 num = r[i];
        for (usize j = 1; j < i; ++j) {
            num += a_prev[j] * r[i - j];
        }
        const f64 k = -num / e;
        if (!keel::is_finite(k)) {
            break;
        }
        a_next[0] = 1.0;
        for (usize j = 1; j < i; ++j) {
            a_next[j] = a_prev[j] + k * a_prev[i - j];
        }
        a_next[i] = k;
        for (usize j = 0; j <= i; ++j) {
            a_prev[j] = a_next[j];
        }
        e *= (1.0 - k * k);
    }

    for (usize i = 0; i <= p; ++i) {
        a_out[i] = a_prev[i];
    }
}

AudioStatus pitch_ncf(ArrayRef<f64> x, f64 fs, f64 min_f, f64 max_f, const FrameSpec& sp, MutArrayRef<f64> f0_out,
                      Allocator* alloc) {
    if (sp.win_len < 2) {
        for (usize f = 0; f < sp.num_frames; ++f) {
            f0_out[f] = 0.0;
        }
        return AudioStatus::kOk;
    }
    const i64 min_lag_i = keel::floor_i64(fs / max_f);
    const i64 ceil_max_i = keel::ceil_i64(fs / min_f);
    const usize min_lag = (min_lag_i > 1) ? as<usize>(min_lag_i) : 1;
    const usize win_m1 = sp.win_len - 1;
    const usize max_lag =
        (ceil_max_i <= 0) ? 0 : ((as<usize>(ceil_max_i) < win_m1) ? as<usize>(ceil_max_i) : win_m1);
    if (max_lag <= min_lag) {
        for (usize f = 0; f < sp.num_frames; ++f) {
            f0_out[f] = 0.0;
        }
        return AudioStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> win(sp.win_len, scratch.allocator());
    hamming_periodic(win);
    Array<f64> frame(sp.win_len, scratch.allocator());
    Array<f64> r_buf(max_lag + 1, scratch.allocator());
    Array<f64> p_buf(max_lag + 1, scratch.allocator());

    for (usize f = 0; f < sp.num_frames; ++f) {
        const usize start = f * sp.hop;
        for (usize i = 0; i < sp.win_len; ++i) {
            frame[i] = x[start + i] * win[i];
        }
        const f64 max_x = frame_max_abs(frame.data(), sp.win_len);
        if (keel::is_nan(max_x)) {
            f0_out[f] = kNaN;
            continue;
        }
        if (max_x > 0.0 && keel::is_finite(max_x)) {
            const f64 inv_x = 1.0 / max_x;
            for (usize i = 0; i < sp.win_len; ++i) {
                frame[i] *= inv_x;
            }
        }
        autocorr_lags(frame.data(), sp.win_len, max_lag + 1, r_buf.data());
        partial_power_lags(frame.data(), sp.win_len, max_lag + 1, p_buf.data());

        const f64 sqrt_total = keel::sqrt(keel::max(0.0, r_buf[0]));
        f64 best_val = -1.0;
        usize best_lag = min_lag;
        for (usize k = min_lag; k <= max_lag; ++k) {
            const f64 denom = sqrt_total * keel::sqrt(keel::max(0.0, p_buf[k])) + 1e-300;
            const f64 g = r_buf[k] / denom;
            if (g > best_val) {
                best_val = g;
                best_lag = k;
            }
        }

        f64 lag = as<f64>(best_lag);
        if (best_lag > min_lag && best_lag < max_lag) {
            const f64 a = r_buf[best_lag - 1] / (sqrt_total * keel::sqrt(keel::max(0.0, p_buf[best_lag - 1])) + 1e-300);
            const f64 b = best_val;
            const f64 c = r_buf[best_lag + 1] / (sqrt_total * keel::sqrt(keel::max(0.0, p_buf[best_lag + 1])) + 1e-300);
            const f64 denom = 2.0 * (2.0 * b - c - a);
            if (denom > 1e-12) {
                f64 shift = -(a - c) / denom;
                if (shift < -0.5) shift = -0.5;
                if (shift > 0.5) shift = 0.5;
                lag = as<f64>(best_lag) + shift;
            }
        }
        f0_out[f] = (lag > 0.0) ? (fs / lag) : 0.0;
    }
    return AudioStatus::kOk;
}

AudioStatus pitch_cep(ArrayRef<f64> x, f64 fs, f64 min_f, f64 max_f, const FrameSpec& sp, MutArrayRef<f64> f0_out,
                      Allocator* alloc) {
    if (sp.win_len > Limits<usize>::kMax / 4) {
        return AudioStatus::kInvalidArgument;
    }
    const usize nfft = next_pow2(2 * sp.win_len - 1);
    if (nfft == 0) {
        return AudioStatus::kInvalidArgument;
    }
    const usize half = nfft / 2 + 1;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> win(sp.win_len, scratch.allocator());
    hamming_periodic(win);

    auto plan_res = fft::FftPlan1D::create(nfft, scratch.allocator());
    if (!plan_res.is_ok()) {
        return AudioStatus::kInvalidArgument;
    }
    const fft::FftPlan1D& plan = plan_res.value();

    Array<f64> frame(nfft, 0.0, scratch.allocator());
    Array<c64> half_spec(half, scratch.allocator());
    Array<c64> log_half(half, scratch.allocator());
    Array<f64> cep(nfft, scratch.allocator());

    i64 q_lo = keel::round_i64(fs / max_f) - 1;
    i64 q_hi = keel::round_i64(fs / min_f) - 1;
    if (q_lo < 0) {
        q_lo = 0;
    }
    const i64 q_max = as<i64>(nfft) - 1;
    if (q_hi > q_max) {
        q_hi = q_max;
    }
    const bool range_valid = (q_lo <= q_hi);
    const f64 log_floor = keel::log(Limits<f64>::kMinPositive);

    for (usize f = 0; f < sp.num_frames; ++f) {
        if (!range_valid) {
            f0_out[f] = 0.0;
            continue;
        }
        const usize base = f * sp.hop;
        for (usize n = 0; n < sp.win_len; ++n) {
            const usize idx = base + n;
            frame[n] = (idx < x.size()) ? (x[idx] * win[n]) : 0.0;
        }
        for (usize n = sp.win_len; n < nfft; ++n) {
            frame[n] = 0.0;
        }

        plan.exec_rfft(frame, half_spec);
        for (usize k = 0; k < half; ++k) {
            const f64 mag = keel::hypot(half_spec[k].real(), half_spec[k].imag());
            const f64 lp = (mag > 0.0) ? (2.0 * keel::log(mag)) : log_floor;
            log_half[k] = c64((lp < log_floor) ? log_floor : lp, 0.0);
        }
        plan.exec_irfft(log_half, cep);

        i64 q_best = q_lo;
        f64 c_best = -Limits<f64>::infinity();
        for (i64 q = q_lo; q <= q_hi; ++q) {
            const f64 cq = cep[as<usize>(q)];
            if (cq > c_best) {
                c_best = cq;
                q_best = q;
            }
        }
        f0_out[f] = fs / as<f64>(q_best + 1);
    }
    return AudioStatus::kOk;
}

AudioStatus pitch_pef(ArrayRef<f64> x, f64 fs, f64 min_f, f64 max_f, const FrameSpec& sp, MutArrayRef<f64> f0_out,
                      Allocator* alloc) {
    constexpr f64 kGamma = 1.5;
    constexpr i64 kK = 10;
    const f64 k_delta = keel::log(1.0058);

    if (sp.win_len > Limits<usize>::kMax / 8) {
        return AudioStatus::kInvalidArgument;
    }
    const usize nfft = next_pow2(4 * sp.win_len);
    if (nfft == 0) {
        return AudioStatus::kInvalidArgument;
    }
    const usize half = nfft / 2;
    const usize pbins = half + 1;

    const i64 j_lo = keel::ceil_i64(keel::log(0.5) / k_delta);
    const i64 j_hi = keel::floor_i64(keel::log(as<f64>(kK) + 0.5) / k_delta);
    const usize taps = (j_hi >= j_lo) ? as<usize>(j_hi - j_lo + 1) : 0;

    if (taps == 0) {
        f64 mid = 0.5 * (min_f + max_f);
        if (mid < min_f) mid = min_f;
        if (mid > max_f) mid = max_f;
        for (usize f = 0; f < sp.num_frames; ++f) {
            f0_out[f] = mid;
        }
        return AudioStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> win(sp.win_len, scratch.allocator());
    hamming_periodic(win);

    auto plan_res = fft::FftPlan1D::create(nfft, scratch.allocator());
    if (!plan_res.is_ok()) {
        return AudioStatus::kInvalidArgument;
    }
    const fft::FftPlan1D& plan = plan_res.value();

    Array<f64> r_tap(taps, scratch.allocator());
    Array<f64> g_tap(taps, scratch.allocator());
    f64 g_sum = 0.0;
    for (usize t = 0; t < taps; ++t) {
        const f64 q = as<f64>(j_lo + as<i64>(t)) * k_delta;
        const f64 r = keel::exp(q);
        const f64 g = keel::log(kGamma - keel::cos(kTau * r));
        r_tap[t] = r;
        g_tap[t] = g;
        g_sum += g;
    }
    const f64 beta = g_sum / as<f64>(taps);
    Array<f64> h_tap(taps, scratch.allocator());
    for (usize t = 0; t < taps; ++t) {
        h_tap[t] = beta - g_tap[t];
    }

    const i64 m_lo = keel::ceil_i64(keel::log(min_f) / k_delta);
    const i64 m_hi = keel::floor_i64(keel::log(max_f) / k_delta);
    const f64 df = fs / as<f64>(nfft);

    Array<f64> frame(nfft, 0.0, scratch.allocator());
    Array<c64> half_spec(pbins, scratch.allocator());
    Array<f64> p_frame(pbins, scratch.allocator());

    for (usize f = 0; f < sp.num_frames; ++f) {
        const usize offset = f * sp.hop;
        for (usize i = 0; i < sp.win_len; ++i) {
            frame[i] = x[offset + i] * win[i];
        }
        const f64 max_x = frame_max_abs(frame.data(), sp.win_len);
        if (keel::is_nan(max_x)) {
            f0_out[f] = kNaN;
            continue;
        }
        if (max_x > 0.0 && keel::is_finite(max_x)) {
            const f64 inv_x = 1.0 / max_x;
            for (usize i = 0; i < sp.win_len; ++i) {
                frame[i] *= inv_x;
            }
        }
        for (usize i = sp.win_len; i < nfft; ++i) {
            frame[i] = 0.0;
        }
        plan.exec_rfft(frame, half_spec);
        for (usize k = 0; k < pbins; ++k) {
            const f64 re = half_spec[k].real();
            const f64 im = half_spec[k].imag();
            p_frame[k] = re * re + im * im;
        }

        f64 best_score = -Limits<f64>::infinity();
        f64 best_f0 = 0.5 * (min_f + max_f);

        for (i64 m = m_lo; m <= m_hi; ++m) {
            const f64 f0 = keel::exp(as<f64>(m) * k_delta);
            f64 score = 0.0;
            for (usize t = 0; t < taps; ++t) {
                const f64 ftap = f0 * r_tap[t];
                f64 p_val = 0.0;
                if (ftap >= 0.0 && ftap <= 0.5 * fs) {
                    const f64 pos = ftap / df;
                    if (keel::is_finite(pos) && pos >= 0.0) {
                        const i64 k0 = keel::floor_i64(pos);
                        if (k0 >= 0 && as<usize>(k0) < half) {
                            const f64 frac = pos - as<f64>(k0);
                            const usize ku = as<usize>(k0);
                            p_val = p_frame[ku] * (1.0 - frac) + p_frame[ku + 1] * frac;
                        } else if (k0 >= 0 && as<usize>(k0) == half) {
                            p_val = p_frame[half];
                        }
                    }
                }
                score += h_tap[t] * p_val;
            }
            if (score > best_score) {
                best_score = score;
                best_f0 = f0;
            }
        }
        if (best_f0 < min_f) best_f0 = min_f;
        if (best_f0 > max_f) best_f0 = max_f;
        f0_out[f] = best_f0;
    }
    return AudioStatus::kOk;
}

AudioStatus pitch_lhs(ArrayRef<f64> x, f64 fs, f64 min_f, f64 max_f, const FrameSpec& sp, MutArrayRef<f64> f0_out,
                      Allocator* alloc) {
    constexpr i64 kH = 5;
    constexpr f64 kMaxSafeFs = 4503599627370496.0; // 2^52
    if (fs >= kMaxSafeFs || max_f > fs / as<f64>(kH) + 1.0) {
        return AudioStatus::kInvalidArgument;
    }
    const i64 nfft_i = keel::round_i64(fs);
    const i64 floor_max_f = keel::floor_i64(max_f);
    if (nfft_i <= 0 || floor_max_f <= 0 || floor_max_f > nfft_i / kH) {
        return AudioStatus::kInvalidArgument;
    }
    const i64 k_max_bin = kH * floor_max_f;

    i64 j_lo_raw = keel::ceil_i64(min_f) - 1;
    i64 j_hi_raw = floor_max_f - 1;
    if (j_lo_raw < 0) {
        j_lo_raw = 0;
    }
    const i64 j_max_cov = (k_max_bin - 1) / kH;
    if (j_hi_raw > j_max_cov) {
        j_hi_raw = j_max_cov;
    }
    if (j_hi_raw < j_lo_raw) {
        return AudioStatus::kInvalidArgument;
    }
    const usize j_lo = as<usize>(j_lo_raw);
    const usize j_hi = as<usize>(j_hi_raw);
    const usize nfft = as<usize>(nfft_i);
    const usize k_bins = as<usize>(k_max_bin);

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> win(sp.win_len, scratch.allocator());
    hamming_periodic(win);

    auto plan_res = fft::FftPlan1D::create(nfft, scratch.allocator());
    if (!plan_res.is_ok()) {
        return AudioStatus::kInvalidArgument;
    }
    const fft::FftPlan1D& plan = plan_res.value();

    Array<f64> frame(nfft, 0.0, scratch.allocator());
    Array<c64> full_spec(nfft, scratch.allocator());
    Array<f64> log_mag(k_bins, scratch.allocator());
    const f64 mag_floor = keel::log(Limits<f64>::kMinPositive);

    for (usize f = 0; f < sp.num_frames; ++f) {
        const usize base = f * sp.hop;
        for (usize n = 0; n < nfft; ++n) {
            f64 s = 0.0;
            if (n < sp.win_len) {
                const usize idx = base + n;
                if (idx < x.size()) {
                    s = x[idx] * win[n];
                }
            }
            frame[n] = s;
        }

        plan.exec_real_to_complex(frame, full_spec);
        for (usize k = 0; k < k_bins; ++k) {
            const f64 m = keel::abs(full_spec[k]);
            log_mag[k] = (m > 0.0) ? keel::log(m) : mag_floor;
        }

        usize best_j = j_lo;
        f64 best_sum = -Limits<f64>::infinity();
        for (usize j = j_lo; j <= j_hi; ++j) {
            f64 sum = 0.0;
            for (usize m = 1; m <= as<usize>(kH); ++m) {
                sum += log_mag[j * m];
            }
            if (sum > best_sum) {
                best_sum = sum;
                best_j = j;
            }
        }

        f64 f0 = as<f64>(best_j) + 1.0;
        if (f0 < min_f) f0 = min_f;
        if (f0 > max_f) f0 = max_f;
        f0_out[f] = f0;
    }
    return AudioStatus::kOk;
}

AudioStatus pitch_srh(ArrayRef<f64> x, f64 fs, f64 min_f, f64 max_f, const FrameSpec& sp, MutArrayRef<f64> f0_out,
                      Allocator* alloc) {
    constexpr usize kLpcOrder = 12;
    constexpr i64 kNharm = 5;
    constexpr f64 kMaxSafeFs = 4503599627370496.0; // 2^52
    if (fs >= kMaxSafeFs) {
        return AudioStatus::kInvalidArgument;
    }
    const i64 nfft_i = keel::round_i64(fs);
    const i64 floor_max_f = keel::floor_i64(max_f);
    if (nfft_i <= 0 || floor_max_f <= 0 || floor_max_f > (nfft_i - 1) / kNharm) {
        for (usize f = 0; f < sp.num_frames; ++f) {
            f0_out[f] = 0.0;
        }
        return AudioStatus::kOk;
    }

    const i64 lo_full = keel::ceil_i64(min_f);
    const i64 hi_full = floor_max_f;
    if (lo_full < 0 || lo_full > hi_full || hi_full >= nfft_i) {
        for (usize f = 0; f < sp.num_frames; ++f) {
            f0_out[f] = 0.0;
        }
        return AudioStatus::kOk;
    }

    const usize nfft = as<usize>(nfft_i);
    if (mul_overflows(sp.num_frames, nfft)) {
        return AudioStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);

    Array<f64> hann(sp.win_len, scratch.allocator());
    if (sp.win_len == 1) {
        hann[0] = 1.0;
    } else {
        const f64 wl_f = as<f64>(sp.win_len);
        for (usize n = 0; n < sp.win_len; ++n) {
            hann[n] = 0.5 * (1.0 - keel::cos(kTau * as<f64>(n) / wl_f));
        }
    }

    auto plan_res = fft::FftPlan1D::create(nfft, scratch.allocator());
    if (!plan_res.is_ok()) {
        return AudioStatus::kInvalidArgument;
    }
    const fft::FftPlan1D& plan = plan_res.value();

    Array<f64> e_spec(sp.num_frames * nfft, scratch.allocator());
    Array<f64> windowed(sp.win_len, scratch.allocator());
    Array<f64> residual(nfft, 0.0, scratch.allocator());
    Array<c64> full_spec(nfft, scratch.allocator());
    f64 lpc_a[kLpcOrder + 1]{};

    for (usize f = 0; f < sp.num_frames; ++f) {
        const usize base = f * sp.hop;
        for (usize n = 0; n < sp.win_len; ++n) {
            const usize idx = base + n;
            const f64 s = (idx < x.size()) ? x[idx] : 0.0;
            windowed[n] = s * hann[n];
        }

        lpc_levinson(windowed.data(), sp.win_len, kLpcOrder, lpc_a, scratch.allocator());

        // FIR inverse filter: residual[n] = sum_{j=0..P} lpc_a[j] * windowed[n - j]
        const usize copy_len = (sp.win_len < nfft) ? sp.win_len : nfft;
        for (usize n = 0; n < copy_len; ++n) {
            f64 acc = 0.0;
            const usize j_max = (n < kLpcOrder) ? n : kLpcOrder;
            for (usize j = 0; j <= j_max; ++j) {
                acc += lpc_a[j] * windowed[n - j];
            }
            residual[n] = acc;
        }
        for (usize n = copy_len; n < nfft; ++n) {
            residual[n] = 0.0;
        }

        plan.exec_real_to_complex(residual, full_spec);
        f64* e_row = e_spec.data() + f * nfft;
        for (usize k = 0; k < nfft; ++k) {
            e_row[k] = keel::abs(full_spec[k]);
        }
    }

    auto srh_score = [&](usize f, i64 cand) -> f64 {
        const f64* e_row = e_spec.data() + f * nfft;
        f64 score = e_row[as<usize>(cand)];
        for (i64 k = 2; k <= kNharm; ++k) {
            const i64 hi = k * cand;
            const i64 mid = keel::round_i64((as<f64>(k) - 0.5) * as<f64>(cand));
            const f64 e_hi = (hi >= 0 && hi < nfft_i) ? e_row[as<usize>(hi)] : 0.0;
            const f64 e_mid = (mid >= 0 && mid < nfft_i) ? e_row[as<usize>(mid)] : 0.0;
            score += e_hi - e_mid;
        }
        return score;
    };

    auto best_cand = [&](usize f, i64 lo, i64 hi) -> i64 {
        i64 best_f = lo;
        f64 best_score = -Limits<f64>::infinity();
        for (i64 cand = lo; cand <= hi; ++cand) {
            const f64 sc = srh_score(f, cand);
            if (sc > best_score) {
                best_score = sc;
                best_f = cand;
            }
        }
        return best_f;
    };

    Array<f64> f0_tmp(sp.num_frames, scratch.allocator());
    f64 sum_pass1 = 0.0;
    for (usize f = 0; f < sp.num_frames; ++f) {
        f0_tmp[f] = as<f64>(best_cand(f, lo_full, hi_full));
        sum_pass1 += f0_tmp[f];
    }
    const f64 f0_mean = sum_pass1 / as<f64>(sp.num_frames);

    if (keel::is_finite(f0_mean) && f0_mean > 0.0) {
        const i64 lo_cand = keel::round_i64(0.5 * f0_mean);
        const i64 hi_cand = keel::round_i64(2.0 * f0_mean);
        const i64 lo_ref = (lo_full > lo_cand) ? lo_full : lo_cand;
        const i64 hi_ref = (hi_full < hi_cand) ? hi_full : hi_cand;
        if (lo_ref <= hi_ref) {
            for (usize f = 0; f < sp.num_frames; ++f) {
                f0_tmp[f] = as<f64>(best_cand(f, lo_ref, hi_ref));
            }
        }
    }

    for (usize f = 0; f < sp.num_frames; ++f) {
        f64 v = f0_tmp[f];
        if (v < min_f) v = min_f;
        if (v > max_f) v = max_f;
        f0_out[f] = v;
    }
    return AudioStatus::kOk;
}

} // namespace

AudioStatus pitch(ArrayRef<f64> x, f64 sample_rate, MutArrayRef<f64> f0_out, f64 min_freq, f64 max_freq,
                  PitchMethod method, Allocator* alloc) {
    if (!(sample_rate > 0.0) || !keel::is_finite(sample_rate) || !(min_freq > 0.0) || !(max_freq > min_freq) ||
        !keel::is_finite(min_freq) || !keel::is_finite(max_freq)) {
        return AudioStatus::kInvalidArgument;
    }
    const FrameSpec sp = make_frame_spec(x.size(), sample_rate, 0.052, 0.042);
    if (sp.num_frames == 0 || sp.win_len == 0) {
        return AudioStatus::kOk;
    }
    if (f0_out.size() < sp.num_frames) {
        return AudioStatus::kInvalidArgument;
    }

    switch (method) {
        case PitchMethod::kNcf:
            return pitch_ncf(x, sample_rate, min_freq, max_freq, sp, f0_out, alloc);
        case PitchMethod::kCep:
            return pitch_cep(x, sample_rate, min_freq, max_freq, sp, f0_out, alloc);
        case PitchMethod::kPef:
            return pitch_pef(x, sample_rate, min_freq, max_freq, sp, f0_out, alloc);
        case PitchMethod::kLhs:
            return pitch_lhs(x, sample_rate, min_freq, max_freq, sp, f0_out, alloc);
        case PitchMethod::kSrh:
            return pitch_srh(x, sample_rate, min_freq, max_freq, sp, f0_out, alloc);
    }
    return AudioStatus::kInvalidArgument;
}

AudioStatus harmonic_ratio(ArrayRef<f64> x, f64 sample_rate, MutArrayRef<f64> hr_out, Allocator* alloc) {
    if (!(sample_rate > 0.0) || !keel::is_finite(sample_rate)) {
        return AudioStatus::kInvalidArgument;
    }
    const FrameSpec sp = make_frame_spec(x.size(), sample_rate, 0.03, 0.02);
    if (sp.num_frames == 0 || sp.win_len == 0) {
        return AudioStatus::kOk;
    }
    if (hr_out.size() < sp.num_frames) {
        return AudioStatus::kInvalidArgument;
    }

    const usize max_lag = sp.win_len - 1;
    const usize m_lag = (max_lag + 1 >= 2) ? (max_lag + 1) : 2;
    const f64 sqrt_eps = keel::sqrt(Limits<f64>::kEpsilon);

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> win(sp.win_len, scratch.allocator());
    hamming_periodic(win);
    Array<f64> frame(sp.win_len, scratch.allocator());
    Array<f64> r_buf(m_lag, scratch.allocator());
    Array<f64> p_buf(m_lag, scratch.allocator());
    Array<f64> gamma(m_lag, scratch.allocator());

    for (usize f = 0; f < sp.num_frames; ++f) {
        const usize start = f * sp.hop;
        for (usize i = 0; i < sp.win_len; ++i) {
            frame[i] = x[start + i] * win[i];
        }
        const f64 max_x = frame_max_abs(frame.data(), sp.win_len);
        if (keel::is_nan(max_x)) {
            hr_out[f] = kNaN;
            continue;
        }
        // Only rescale when squaring would overflow f64 (> 1e150), preserving
        // exact MATLAB sqrt(eps) floor semantics for normal and quiet frames.
        const bool rescale = keel::is_finite(max_x) && max_x > 1e150;
        const f64 inv_x = rescale ? (1.0 / max_x) : 1.0;
        if (rescale) {
            for (usize i = 0; i < sp.win_len; ++i) {
                frame[i] *= inv_x;
            }
        }
        autocorr_lags(frame.data(), sp.win_len, m_lag, r_buf.data());
        partial_power_lags(frame.data(), sp.win_len, m_lag, p_buf.data());

        const f64 sqrt_total = keel::sqrt(keel::max(0.0, r_buf[0]));
        const f64 eps_term = rescale ? 0.0 : sqrt_eps;
        for (usize k = 0; k < m_lag; ++k) {
            const f64 denom = sqrt_total * keel::sqrt(keel::max(0.0, p_buf[k])) + eps_term;
            gamma[k] = (denom > 0.0) ? (r_buf[k] / denom) : 0.0;
        }

        usize low_edge = max_lag;
        const f64 r0 = r_buf[0];
        const i8 prev_sign = (r0 > 0.0) ? as<i8>(1) : (r0 < 0.0 ? as<i8>(-1) : as<i8>(0));
        for (usize k = 1; k < m_lag; ++k) {
            const i8 s = (r_buf[k] > 0.0) ? as<i8>(1)
                                          : (r_buf[k] < 0.0 ? as<i8>(-1) : as<i8>(0));
            if (s != prev_sign) {
                low_edge = k + 1;
                break;
            }
        }
        if (low_edge < 1) {
            low_edge = 1;
        }
        for (usize k = 0; k < low_edge && k < m_lag; ++k) {
            gamma[k] = 0.0;
        }

        usize peak_idx = low_edge;
        f64 peak_val = 0.0;
        for (usize k = low_edge; k < m_lag; ++k) {
            if (gamma[k] > peak_val) {
                peak_val = gamma[k];
                peak_idx = k;
            }
        }

        f64 refined = peak_val;
        if (peak_idx > 0 && peak_idx + 1 < m_lag) {
            const f64 a = gamma[peak_idx - 1];
            const f64 b = gamma[peak_idx];
            const f64 c = gamma[peak_idx + 1];
            const f64 denom_s = 2.0 * (2.0 * b - c - a);
            if (denom_s > 1e-12) {
                const f64 s = (c - a) / denom_s;
                refined = b - 0.25 * (a - c) * s;
            }
        }
        if (refined < 0.0) refined = 0.0;
        if (refined > 1.0) refined = 1.0;
        hr_out[f] = refined;
    }
    return AudioStatus::kOk;
}

} // namespace keel::audio
