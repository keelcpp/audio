#include "keel/audio.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::audio {

void hamming_periodic(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    const f64 nf = as<f64>(n);
    for (usize i = 0; i < n; ++i) {
        out[i] = 0.54 - 0.46 * keel::cos(kTau * as<f64>(i) / nf);
    }
}

AudioStatus mel_filterbank(f64 sample_rate, usize num_bands, usize fft_len,
                           MutArrayRef<f64> fb_col_major,
                           MutArrayRef<f64> center_freqs_hz) noexcept {
    if (!(sample_rate > 0.0) || !keel::is_finite(sample_rate) || fft_len == 0) {
        return AudioStatus::kInvalidArgument;
    }
    if (num_bands == 0) {
        return AudioStatus::kOk;
    }
    const usize h = fft_len / 2 + 1;
    if (fb_col_major.size() < num_bands * h || center_freqs_hz.size() < num_bands) {
        return AudioStatus::kInvalidArgument;
    }

    const usize total = num_bands * h;
    for (usize i = 0; i < total; ++i) {
        fb_col_major[i] = 0.0;
    }

    const f64 mel_max = hz_to_mel(sample_rate * 0.5);
    const f64 denom_bands = as<f64>(num_bands + 1);
    const f64 fft_f = as<f64>(fft_len);

    for (usize b = 0; b < num_bands; ++b) {
        const f64 m_lo = as<f64>(b) / denom_bands * mel_max;
        const f64 m_ce = as<f64>(b + 1) / denom_bands * mel_max;
        const f64 m_hi = as<f64>(b + 2) / denom_bands * mel_max;
        const f64 f_lo = mel_to_hz(m_lo);
        const f64 f_ce = mel_to_hz(m_ce);
        const f64 f_hi = mel_to_hz(m_hi);
        center_freqs_hz[b] = f_ce;

        for (usize k = 0; k < h; ++k) {
            const f64 f = as<f64>(k) * sample_rate / fft_f;
            f64 val = 0.0;
            if (f >= f_lo && f <= f_ce && f_ce > f_lo) {
                val = (f - f_lo) / (f_ce - f_lo);
            } else if (f > f_ce && f <= f_hi && f_hi > f_ce) {
                val = (f_hi - f) / (f_hi - f_ce);
            }
            fb_col_major[b + k * num_bands] = val;
        }
        const f64 bw = f_hi - f_lo;
        if (bw > 0.0) {
            const f64 inv = 2.0 / bw;
            for (usize k = 0; k < h; ++k) {
                fb_col_major[b + k * num_bands] *= inv;
            }
        }
    }
    return AudioStatus::kOk;
}

AudioStatus audio_delta(ArrayRef<f64> x, usize window_length, MutArrayRef<f64> y, ArrayRef<f64> zi,
                        MutArrayRef<f64> zf, Allocator* alloc) {
    if (window_length < 3 || (window_length & 1u) == 0u) {
        return AudioStatus::kInvalidArgument;
    }
    const usize n = x.size();
    if (y.size() < n) {
        return AudioStatus::kInvalidArgument;
    }
    const usize state_len = window_length - 1;
    if (!zi.empty() && zi.size() < state_len) {
        return AudioStatus::kInvalidArgument;
    }
    if (!zf.empty() && zf.size() < state_len) {
        return AudioStatus::kInvalidArgument;
    }

    const i64 m = as<i64>(window_length / 2);
    f64 denom = 0.0;
    for (i64 i = 1; i <= m; ++i) {
        denom += as<f64>(i * i);
    }
    if (denom <= 0.0) denom = 1.0;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> b(window_length, scratch.allocator());
    for (usize k = 0; k < window_length; ++k) {
        b[k] = as<f64>(m - as<i64>(k)) / denom;
    }
    Array<f64> z(state_len, 0.0, scratch.allocator());
    if (!zi.empty()) {
        for (usize i = 0; i < state_len; ++i) {
            z[i] = zi[i];
        }
    }

    for (usize i = 0; i < n; ++i) {
        const f64 xv = x[i];
        y[i] = b[0] * xv + z[0];
        for (usize k = 0; k + 1 < state_len; ++k) {
            z[k] = b[k + 1] * xv + z[k + 1];
        }
        z[state_len - 1] = b[window_length - 1] * xv;
    }

    if (!zf.empty()) {
        for (usize i = 0; i < state_len; ++i) {
            zf[i] = z[i];
        }
    }
    return AudioStatus::kOk;
}

} // namespace keel::audio
