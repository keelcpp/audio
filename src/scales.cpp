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

constexpr f64 kErbScaleLn = 1000.0 / (24.673 * 4.368);

[[nodiscard]] bool overlaps_forward(const f64* in_ptr, const f64* out_ptr, usize n) noexcept {
    if (n == 0 || in_ptr == nullptr || out_ptr == nullptr) {
        return false;
    }
    const usize in_addr = reinterpret_cast<usize>(in_ptr);
    const usize out_addr = reinterpret_cast<usize>(out_ptr);
    const usize byte_span = n * sizeof(f64);
    return out_addr > in_addr && (out_addr - in_addr) < byte_span;
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
    if (keel::is_nan(s)) {
        return kNaN;
    }
    if (s < 0.0) {
        return kNaN;
    }
    if (s <= kTab5Sone[0]) {
        return 0.0;
    }
    if (s >= kTab5Sone[kTab5N - 1]) {
        return linear_extrap(kTab5Sone, kTab5Phon, kTab5N, s);
    }
    const f64 q[1] = {s};
    f64 y[1] = {0.0};
    (void)keel::interp::ppval(kTab5Sone, coefs_s2p, kTab5Pieces, 4, q, y);
    return (y[0] < 0.0) ? 0.0 : y[0];
}

[[nodiscard]] f64 eval_phon2sone_532_2(f64 p, ArrayRef<f64> coefs_s2p) noexcept {
    if (keel::is_nan(p) || p < 0.0) {
        return kNaN;
    }
    if (p == 0.0) {
        return kTab5Sone[0];
    }
    if (p >= kTab5Phon[kTab5N - 1]) {
        return linear_extrap(kTab5Phon, kTab5Sone, kTab5N, p);
    }

    // Locate the unique strictly monotonic table interval [kTab5Phon[idx], kTab5Phon[idx + 1]].
    usize idx = 0;
    while (idx + 1 < kTab5Pieces && kTab5Phon[idx + 1] < p) {
        ++idx;
    }
    if (p == kTab5Phon[idx]) {
        return kTab5Sone[idx];
    }
    if (p == kTab5Phon[idx + 1]) {
        return kTab5Sone[idx + 1];
    }

    f64 lo = kTab5Sone[idx];
    f64 hi = kTab5Sone[idx + 1];
    for (usize it = 0; it < 80; ++it) {
        const f64 mid = 0.5 * (lo + hi);
        if (hi - lo <= 1e-14 * keel::max(1.0, mid)) {
            return mid;
        }
        const f64 f_mid = eval_sone2phon_532_2(mid, coefs_s2p) - p;
        if (f_mid == 0.0) {
            return mid;
        }
        if (f_mid > 0.0) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return 0.5 * (lo + hi);
}

template <typename Fn>
AudioStatus apply_unary(ArrayRef<f64> in, MutArrayRef<f64> out, Fn fn) noexcept {
    const usize n = in.size();
    if (out.size() < n) {
        return AudioStatus::kInvalidArgument;
    }
    if (overlaps_forward(in.data(), out.data(), n)) {
        for (usize i = n; i > 0; --i) {
            const usize idx = i - 1;
            out[idx] = fn(in[idx]);
        }
    } else {
        for (usize i = 0; i < n; ++i) {
            out[i] = fn(in[i]);
        }
    }
    return AudioStatus::kOk;
}

} // namespace

f64 hz_to_mel(f64 hz) noexcept {
    return (2595.0 / kLn10) * keel::log1p(hz / 700.0);
}

f64 mel_to_hz(f64 mel) noexcept {
    return 700.0 * keel::expm1(mel * (kLn10 / 2595.0));
}

f64 hz_to_bark(f64 hz) noexcept {
    if (keel::is_nan(hz)) {
        return kNaN;
    }
    f64 bark = 0.0;
    if (hz > 1e100) {
        bark = 26.81 / (1960.0 / hz + 1.0) - 0.53;
    } else {
        bark = 26.81 * hz / (1960.0 + hz) - 0.53;
    }
    if (bark < 2.0) {
        bark = 0.85 * bark + 0.3;
    } else if (bark > 20.1) {
        bark = 1.22 * bark - 4.422;
    }
    return bark;
}

f64 bark_to_hz(f64 bark) noexcept {
    if (keel::is_nan(bark)) {
        return kNaN;
    }
    f64 b = bark;
    if (b < 2.0) {
        b = (b - 0.3) / 0.85;
    } else if (b > 20.1) {
        b = (b + 0.22 * 20.1) / 1.22;
    }
    return 1960.0 * (b + 0.53) / (26.28 - b);
}

f64 hz_to_erb(f64 hz) noexcept {
    return kErbScaleLn * keel::log1p(0.004368 * hz);
}

f64 erb_to_hz(f64 erb) noexcept {
    return keel::expm1(erb / kErbScaleLn) / 0.004368;
}

f64 phon_to_sone(f64 phon, bool iso532_2, Allocator* alloc) {
    if (keel::is_nan(phon) || phon < 0.0) {
        return kNaN;
    }
    if (!iso532_2) {
        if (phon < 40.0) {
            return keel::pow(phon / 40.0, 1.0 / 0.35);
        }
        return keel::pow(2.0, phon / 10.0 - 4.0);
    }
    f64 coefs_s2p[kTab5Pieces * 4]{};
    (void)keel::interp::pchip_pp_coefs(kTab5Sone, kTab5Phon, coefs_s2p, alloc);
    return eval_phon2sone_532_2(phon, coefs_s2p);
}

f64 sone_to_phon(f64 sone, bool iso532_2, Allocator* alloc) {
    if (keel::is_nan(sone) || sone < 0.0) {
        return kNaN;
    }
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
    return apply_unary(in, out, [](f64 v) noexcept { return hz_to_mel(v); });
}

AudioStatus mel_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    return apply_unary(in, out, [](f64 v) noexcept { return mel_to_hz(v); });
}

AudioStatus hz_to_bark(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    return apply_unary(in, out, [](f64 v) noexcept { return hz_to_bark(v); });
}

AudioStatus bark_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    return apply_unary(in, out, [](f64 v) noexcept { return bark_to_hz(v); });
}

AudioStatus hz_to_erb(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    return apply_unary(in, out, [](f64 v) noexcept { return hz_to_erb(v); });
}

AudioStatus erb_to_hz(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    return apply_unary(in, out, [](f64 v) noexcept { return erb_to_hz(v); });
}

AudioStatus phon_to_sone(ArrayRef<f64> in, MutArrayRef<f64> out, bool iso532_2, Allocator* alloc) {
    if (out.size() < in.size()) {
        return AudioStatus::kInvalidArgument;
    }
    if (!iso532_2) {
        return apply_unary(in, out, [](f64 v) noexcept { return phon_to_sone(v, false, nullptr); });
    }
    f64 coefs_s2p[kTab5Pieces * 4]{};
    (void)keel::interp::pchip_pp_coefs(kTab5Sone, kTab5Phon, coefs_s2p, alloc);
    const ArrayRef<f64> coefs_ref(coefs_s2p, kTab5Pieces * 4);
    return apply_unary(in, out, [coefs_ref](f64 v) noexcept { return eval_phon2sone_532_2(v, coefs_ref); });
}

AudioStatus sone_to_phon(ArrayRef<f64> in, MutArrayRef<f64> out, bool iso532_2, Allocator* alloc) {
    if (out.size() < in.size()) {
        return AudioStatus::kInvalidArgument;
    }
    if (!iso532_2) {
        return apply_unary(in, out, [](f64 v) noexcept { return sone_to_phon(v, false, nullptr); });
    }
    f64 coefs_s2p[kTab5Pieces * 4]{};
    (void)keel::interp::pchip_pp_coefs(kTab5Sone, kTab5Phon, coefs_s2p, alloc);
    const ArrayRef<f64> coefs_ref(coefs_s2p, kTab5Pieces * 4);
    return apply_unary(in, out, [coefs_ref](f64 v) noexcept { return eval_sone2phon_532_2(v, coefs_ref); });
}

} // namespace keel::audio
