#include "keel/audio.h"

#include "keel/interp.h"
#include "keel/math.h"

namespace keel::audio {

namespace {

constexpr usize kTab5N = 28;
constexpr usize kTab5Pieces = kTab5N - 1;

constexpr f64 kTab5Phon[kTab5N] = {
    0.000, 2.200, 4.000, 5.000, 7.500, 10.00, 15.00, 20.00, 25.0,  30.0,  35.0,  40.0,  45.0,  50.0,
    55.0,  60.0,  65.0,  70.0,  75.0,  80.0,  85.0,  90.0,  95.0,  100.0, 105.0, 110.0, 115.0, 120.0,
};

constexpr f64 kTab5Sone[kTab5N] = {
    0.001, 0.004, 0.008, 0.010, 0.019, 0.031, 0.073, 0.146, 0.26,  0.43,  0.67,  1.00,  1.46,  2.09,
    2.96,  4.14,  5.77,  8.04,  11.2,  15.8,  22.7,  32.9,  47.7,  69.6,  102.0, 151.0, 225.0, 337.6,
};

[[nodiscard]] f64 erb_scale() noexcept {
    return keel::log(10.0) * 1000.0 / (24.673 * 4.368);
}

[[nodiscard]] f64 linear_extrap(const f64* xs, const f64* ys, usize n, f64 x) noexcept {
    const f64 x0 = xs[n - 2];
    const f64 y0 = ys[n - 2];
    const f64 x1 = xs[n - 1];
    const f64 y1 = ys[n - 1];
    const f64 slope = (x1 - x0 != 0.0) ? (y1 - y0) / (x1 - x0) : 0.0;
    return y1 + slope * (x - x1);
}

[[nodiscard]] f64 eval_sone2phon_532_2(f64 s, ArrayRef<f64> coefs_s2p) noexcept {
    f64 phon_val = 0.0;
    if (s > kTab5Sone[kTab5N - 1]) {
        phon_val = linear_extrap(kTab5Sone, kTab5Phon, kTab5N, s);
    } else {
        const f64 q[1] = {s};
        f64 y[1] = {0.0};
        (void)keel::interp::ppval(kTab5Sone, coefs_s2p, kTab5Pieces, 4, q, y);
        phon_val = y[0];
    }
    return (phon_val < 0.0) ? 0.0 : phon_val;
}

[[nodiscard]] f64 eval_phon2sone_532_2(f64 p, ArrayRef<f64> coefs_p2s,
                                       ArrayRef<f64> coefs_s2p) noexcept {
    const f64 p_capped = (p > 144.0) ? 144.0 : p;
    const f64 q[1] = {p_capped};
    f64 guess_buf[1] = {0.0};
    (void)keel::interp::ppval(kTab5Phon, coefs_p2s, kTab5Pieces, 4, q, guess_buf);
    const f64 guess = guess_buf[0];
    if (guess <= 0.0) {
        return guess;
    }

    f64 lo = guess * 0.5;
    f64 hi = guess * 1.5;
    f64 f_lo = eval_sone2phon_532_2(lo, coefs_s2p) - p_capped;
    f64 f_hi = eval_sone2phon_532_2(hi, coefs_s2p) - p_capped;
    usize expand = 0;
    while (f_lo * f_hi > 0.0 && expand < 30) {
        lo *= 0.5;
        hi *= 1.5;
        f_lo = eval_sone2phon_532_2(lo, coefs_s2p) - p_capped;
        f_hi = eval_sone2phon_532_2(hi, coefs_s2p) - p_capped;
        ++expand;
    }
    if (f_lo * f_hi > 0.0) {
        return guess;
    }
    for (usize it = 0; it < 80; ++it) {
        const f64 mid = 0.5 * (lo + hi);
        if (hi - lo < 1e-12 * keel::max(1.0, keel::abs(mid))) {
            return mid;
        }
        const f64 f_mid = eval_sone2phon_532_2(mid, coefs_s2p) - p_capped;
        if (f_mid == 0.0) {
            return mid;
        }
        if (f_lo * f_mid < 0.0) {
            hi = mid;
            f_hi = f_mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    return 0.5 * (lo + hi);
}

} // namespace

f64 hz_to_mel(f64 hz) noexcept {
    return 2595.0 * keel::log10(1.0 + hz / 700.0);
}

f64 mel_to_hz(f64 mel) noexcept {
    return 700.0 * (keel::pow(10.0, mel / 2595.0) - 1.0);
}

f64 hz_to_bark(f64 hz) noexcept {
    f64 bark = 26.81 * hz / (1960.0 + hz) - 0.53;
    if (bark < 2.0) {
        bark = 0.85 * bark + 0.3;
    } else if (bark > 20.1) {
        bark = 1.22 * bark - 4.422;
    }
    return bark;
}

f64 bark_to_hz(f64 bark) noexcept {
    f64 b = bark;
    if (b < 2.0) {
        b = (b - 0.3) / 0.85;
    } else if (b > 20.1) {
        b = (b + 0.22 * 20.1) / 1.22;
    }
    return 1960.0 * (b + 0.53) / (26.28 - b);
}

f64 hz_to_erb(f64 hz) noexcept {
    return erb_scale() * keel::log10(1.0 + 0.004368 * hz);
}

f64 erb_to_hz(f64 erb) noexcept {
    return (keel::pow(10.0, erb / erb_scale()) - 1.0) / 0.004368;
}

f64 phon_to_sone(f64 phon, bool iso532_2, Allocator* alloc) {
    if (!iso532_2) {
        if (phon < 40.0) {
            return keel::pow(phon / 40.0, 1.0 / 0.35);
        }
        return keel::pow(2.0, phon / 10.0 - 4.0);
    }
    f64 coefs_p2s[kTab5Pieces * 4]{};
    f64 coefs_s2p[kTab5Pieces * 4]{};
    (void)keel::interp::pchip_pp_coefs(kTab5Phon, kTab5Sone, coefs_p2s, alloc);
    (void)keel::interp::pchip_pp_coefs(kTab5Sone, kTab5Phon, coefs_s2p, alloc);
    return eval_phon2sone_532_2(phon, coefs_p2s, coefs_s2p);
}

f64 sone_to_phon(f64 sone, bool iso532_2, Allocator* alloc) {
    if (!iso532_2) {
        if (sone < 1.0) {
            return 40.0 * keel::pow(sone, 0.35);
        }
        return 40.0 + 10.0 * keel::log2(sone);
    }
    f64 coefs_s2p[kTab5Pieces * 4]{};
    (void)keel::interp::pchip_pp_coefs(kTab5Sone, kTab5Phon, coefs_s2p, alloc);
    return eval_sone2phon_532_2(sone, coefs_s2p);
}

AudioStatus hz_to_mel(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    if (out.size() < in.size()) return AudioStatus::kInvalidArgument;
    for (usize i = 0; i < in.size(); ++i) out[i] = hz_to_mel(in[i]);
    return AudioStatus::kOk;
}

AudioStatus mel_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    if (out.size() < in.size()) return AudioStatus::kInvalidArgument;
    for (usize i = 0; i < in.size(); ++i) out[i] = mel_to_hz(in[i]);
    return AudioStatus::kOk;
}

AudioStatus hz_to_bark(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    if (out.size() < in.size()) return AudioStatus::kInvalidArgument;
    for (usize i = 0; i < in.size(); ++i) out[i] = hz_to_bark(in[i]);
    return AudioStatus::kOk;
}

AudioStatus bark_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    if (out.size() < in.size()) return AudioStatus::kInvalidArgument;
    for (usize i = 0; i < in.size(); ++i) out[i] = bark_to_hz(in[i]);
    return AudioStatus::kOk;
}

AudioStatus hz_to_erb(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    if (out.size() < in.size()) return AudioStatus::kInvalidArgument;
    const f64 scale = erb_scale();
    for (usize i = 0; i < in.size(); ++i) {
        out[i] = scale * keel::log10(1.0 + 0.004368 * in[i]);
    }
    return AudioStatus::kOk;
}

AudioStatus erb_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    if (out.size() < in.size()) return AudioStatus::kInvalidArgument;
    const f64 scale = erb_scale();
    for (usize i = 0; i < in.size(); ++i) {
        out[i] = (keel::pow(10.0, in[i] / scale) - 1.0) / 0.004368;
    }
    return AudioStatus::kOk;
}

AudioStatus phon_to_sone(ArrayRef<f64> in, MutArrayRef<f64> out, bool iso532_2, Allocator* alloc) {
    if (out.size() < in.size()) return AudioStatus::kInvalidArgument;
    if (!iso532_2) {
        for (usize i = 0; i < in.size(); ++i) {
            out[i] = phon_to_sone(in[i], false, nullptr);
        }
        return AudioStatus::kOk;
    }
    f64 coefs_p2s[kTab5Pieces * 4]{};
    f64 coefs_s2p[kTab5Pieces * 4]{};
    (void)keel::interp::pchip_pp_coefs(kTab5Phon, kTab5Sone, coefs_p2s, alloc);
    (void)keel::interp::pchip_pp_coefs(kTab5Sone, kTab5Phon, coefs_s2p, alloc);
    for (usize i = 0; i < in.size(); ++i) {
        out[i] = eval_phon2sone_532_2(in[i], coefs_p2s, coefs_s2p);
    }
    return AudioStatus::kOk;
}

AudioStatus sone_to_phon(ArrayRef<f64> in, MutArrayRef<f64> out, bool iso532_2, Allocator* alloc) {
    if (out.size() < in.size()) return AudioStatus::kInvalidArgument;
    if (!iso532_2) {
        for (usize i = 0; i < in.size(); ++i) {
            out[i] = sone_to_phon(in[i], false, nullptr);
        }
        return AudioStatus::kOk;
    }
    f64 coefs_s2p[kTab5Pieces * 4]{};
    (void)keel::interp::pchip_pp_coefs(kTab5Sone, kTab5Phon, coefs_s2p, alloc);
    for (usize i = 0; i < in.size(); ++i) {
        out[i] = eval_sone2phon_532_2(in[i], coefs_s2p);
    }
    return AudioStatus::kOk;
}

} // namespace keel::audio
