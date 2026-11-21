#include "keel/audio.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/complex.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::audio {

namespace {

struct FrameGeometry {
    usize win_len = 0;
    usize overlap = 0;
    usize hop = 1;
    usize num_frames = 0;
};

FrameGeometry compute_frame_geometry(usize num_samples, f64 sample_rate, f64 win_sec, f64 overlap_sec) noexcept {
    FrameGeometry g{};
    if (!(sample_rate > 0.0) || !keel::is_finite(sample_rate) || !(win_sec > 0.0) || !keel::is_finite(win_sec) ||
        !(overlap_sec >= 0.0) || !keel::is_finite(overlap_sec)) {
        return g;
    }
    const i64 w_i = keel::round_i64(sample_rate * win_sec);
    const i64 o_i = keel::round_i64(sample_rate * overlap_sec);
    if (w_i <= 0) {
        return g;
    }
    g.win_len = static_cast<usize>(w_i);
    g.overlap = (o_i > 0) ? static_cast<usize>(o_i) : 0;
    if (g.overlap >= g.win_len) {
        g.overlap = g.win_len - 1;
    }
    g.hop = (g.win_len > g.overlap) ? (g.win_len - g.overlap) : 1;
    g.num_frames = (num_samples >= g.win_len) ? ((num_samples - g.win_len) / g.hop + 1) : 0;
    return g;
}

f64 rectify_value(f64 v, CepstralRectification rect, f64 log_floor) noexcept {
    switch (rect) {
        case CepstralRectification::kLog: {
            const f64 r = (v > 0.0) ? keel::log10(v) : log_floor;
            return (r < log_floor) ? log_floor : r;
        }
        case CepstralRectification::kCubicRoot:
            return keel::cbrt(v);
        case CepstralRectification::kNone:
            return v;
    }
    return v;
}

void build_dct2_basis(usize num_bands, usize num_coeffs, MutArrayRef<f64> basis_col_major) noexcept {
    const f64 l_f = static_cast<f64>(num_bands);
    const f64 a = keel::sqrt(1.0 / l_f);
    const f64 b = keel::sqrt(2.0 / l_f);
    for (usize l = 0; l < num_bands; ++l) {
        basis_col_major[0 + l * num_coeffs] = a;
    }
    for (usize l = 0; l < num_bands; ++l) {
        const f64 lc = static_cast<f64>(l) + 0.5;
        for (usize n = 1; n < num_coeffs; ++n) {
            basis_col_major[n + l * num_coeffs] = b * keel::cos(kPi * static_cast<f64>(n) * lc / l_f);
        }
    }
}

AudioStatus compute_deltas(ArrayRef<f64> coeffs_col_major, usize num_frames, usize num_cols,
                           MutArrayRef<f64> delta_col_major, MutArrayRef<f64> delta_delta_col_major,
                           Allocator* scratch_alloc) {
    const usize total = num_frames * num_cols;
    const bool want_d = !delta_col_major.empty();
    const bool want_dd = !delta_delta_col_major.empty();
    if (!want_d && !want_dd) {
        return AudioStatus::kOk;
    }
    if (want_d && delta_col_major.size() < total) {
        return AudioStatus::kInvalidArgument;
    }
    if (want_dd && delta_delta_col_major.size() < total) {
        return AudioStatus::kInvalidArgument;
    }

    Array<f64> temp_d(want_d ? 0 : total, scratch_alloc);
    MutArrayRef<f64> d_buf = want_d ? delta_col_major : MutArrayRef<f64>(temp_d.data(), total);

    for (usize c = 0; c < num_cols; ++c) {
        const ArrayRef<f64> src_col(coeffs_col_major.data() + c * num_frames, num_frames);
        const MutArrayRef<f64> dst_col(d_buf.data() + c * num_frames, num_frames);
        const AudioStatus st = audio_delta(src_col, 9, dst_col, {}, {}, scratch_alloc);
        if (st != AudioStatus::kOk) {
            return st;
        }
    }

    if (want_dd) {
        for (usize c = 0; c < num_cols; ++c) {
            const ArrayRef<f64> src_col(d_buf.data() + c * num_frames, num_frames);
            const MutArrayRef<f64> dst_col(delta_delta_col_major.data() + c * num_frames, num_frames);
            const AudioStatus st = audio_delta(src_col, 9, dst_col, {}, {}, scratch_alloc);
            if (st != AudioStatus::kOk) {
                return st;
            }
        }
    }
    return AudioStatus::kOk;
}

} // namespace

usize audio_frame_count(usize num_samples, f64 sample_rate, f64 win_sec, f64 overlap_sec) noexcept {
    return compute_frame_geometry(num_samples, sample_rate, win_sec, overlap_sec).num_frames;
}

void slaney_mel_band_edges(MutArrayRef<f64> edges) noexcept {
    const usize num_edges = edges.size();
    if (num_edges == 0) {
        return;
    }
    constexpr f64 kFactor = 133.33333333333333;
    const usize lin_end = (num_edges < 13) ? num_edges : 13;
    for (usize i = 0; i < lin_end; ++i) {
        edges[i] = kFactor + (kFactor * 0.5) * static_cast<f64>(i);
    }
    for (usize i = lin_end; i < num_edges; ++i) {
        edges[i] = edges[i - 1] * 1.0711703;
    }
}

AudioStatus slaney_mel_filterbank(f64 sample_rate, usize fft_len, ArrayRef<f64> edges,
                                  MutArrayRef<f64> fb_col_major) noexcept {
    if (!(sample_rate > 0.0) || !keel::is_finite(sample_rate) || fft_len == 0 || edges.size() < 3) {
        return AudioStatus::kInvalidArgument;
    }
    const usize h = fft_len / 2 + 1;
    const usize num_bands = edges.size() - 2;
    if (fb_col_major.size() < h * num_bands) {
        return AudioStatus::kInvalidArgument;
    }

    const usize total = h * num_bands;
    for (usize i = 0; i < total; ++i) {
        fb_col_major[i] = 0.0;
    }

    const f64 bin_to_hz = sample_rate / static_cast<f64>(fft_len);
    for (usize k = 0; k < num_bands; ++k) {
        const f64 left = edges[k];
        const f64 centre = edges[k + 1];
        const f64 right = edges[k + 2];

        const f64 rise_w = centre - left;
        const f64 fall_w = right - centre;
        const f64 base_w = right - left;
        if (rise_w <= 0.0 || fall_w <= 0.0 || base_w <= 0.0) {
            continue;
        }

        const f64 norm = 2.0 / base_w;
        f64* col = fb_col_major.data() + k * h;
        for (usize j = 0; j < h; ++j) {
            const f64 f = static_cast<f64>(j) * bin_to_hz;
            const f64 rise = (f - left) / rise_w;
            const f64 fall = (right - f) / fall_w;
            f64 w = (rise < fall) ? rise : fall;
            if (w < 0.0) {
                w = 0.0;
            }
            col[j] = w * norm;
        }
    }
    return AudioStatus::kOk;
}

AudioStatus gammatone_filterbank(f64 sample_rate, ArrayRef<f64> center_freqs_hz, usize fft_len,
                                 MutArrayRef<f64> fb_row_major, bool bandwidth_normalize) noexcept {
    if (!(sample_rate > 0.0) || !keel::is_finite(sample_rate) || fft_len == 0) {
        return AudioStatus::kInvalidArgument;
    }
    const usize num_bands = center_freqs_hz.size();
    if (num_bands == 0) {
        return AudioStatus::kOk;
    }
    const usize h = fft_len / 2 + 1;
    if (fb_row_major.size() < num_bands * h) {
        return AudioStatus::kInvalidArgument;
    }

    const f64 t = 1.0 / sample_rate;
    constexpr f64 kEarQ = 9.26449;
    constexpr f64 kMinBw = 24.7;
    const f64 two_pow_1p5 = keel::pow(2.0, 1.5);
    const f64 s32a = keel::sqrt(3.0 + two_pow_1p5);
    const f64 s32b = keel::sqrt(3.0 - two_pow_1p5);
    const c64 j(0.0, 1.0);

    for (usize i = 0; i < num_bands; ++i) {
        const f64 cf = center_freqs_hz[i];
        const f64 erb = cf / kEarQ + kMinBw;
        const f64 b_rad = 1.019 * kTau * erb;

        const f64 cos_arg = kTau * cf * t;
        const f64 cos_t = keel::cos(cos_arg);
        const f64 sin_t = keel::sin(cos_arg);
        const f64 exp_bt = keel::exp(b_rad * t);
        const f64 exp_m2bt = keel::exp(-2.0 * b_rad * t);

        const f64 a0 = t;
        const f64 a11 = -(2.0 * t * cos_t / exp_bt + 2.0 * s32a * t * sin_t / exp_bt) * 0.5;
        const f64 a12 = -(2.0 * t * cos_t / exp_bt - 2.0 * s32a * t * sin_t / exp_bt) * 0.5;
        const f64 a13 = -(2.0 * t * cos_t / exp_bt + 2.0 * s32b * t * sin_t / exp_bt) * 0.5;
        const f64 a14 = -(2.0 * t * cos_t / exp_bt - 2.0 * s32b * t * sin_t / exp_bt) * 0.5;
        const f64 b0 = 1.0;
        const f64 b1 = -2.0 * cos_t / exp_bt;
        const f64 b2 = exp_m2bt;

        const c64 exp4j = keel::exp(c64(0.0, 2.0 * cos_arg));
        const c64 exp_2j = keel::exp(c64(0.0, cos_arg)) / exp_bt;
        const c64 term1 = -2.0 * exp4j * t + 2.0 * exp_2j * t * (cos_t - s32b * sin_t);
        const c64 term2 = -2.0 * exp4j * t + 2.0 * exp_2j * t * (cos_t + s32b * sin_t);
        const c64 term3 = -2.0 * exp4j * t + 2.0 * exp_2j * t * (cos_t - s32a * sin_t);
        const c64 term4 = -2.0 * exp4j * t + 2.0 * exp_2j * t * (cos_t + s32a * sin_t);
        const c64 denom_simple = -2.0 / keel::exp(2.0 * b_rad * t) - 2.0 * exp4j + 2.0 * (c64(1.0, 0.0) + exp4j) / exp_bt;
        const c64 denom2 = denom_simple * denom_simple;
        const c64 denom4 = denom2 * denom2;
        const f64 gain = keel::abs((term1 * term2 * term3 * term4) / denom4);

        const f64 b0_1 = a0 / gain;
        const f64 b1_1 = a11 / gain;

        f64 scale = 1.0;
        if (bandwidth_normalize) {
            const f64 bw = 1.019 * 24.7 * (0.00437 * cf + 1.0);
            scale = (bw > 0.0) ? (2.0 / bw) : 0.0;
        }

        f64* row = fb_row_major.data() + i * h;
        const f64 nfft_f = static_cast<f64>(fft_len);
        for (usize k = 0; k < h; ++k) {
            const f64 w = -kTau * static_cast<f64>(k) / nfft_f;
            const c64 zinv = keel::exp(j * w);
            const c64 zinv2 = zinv * zinv;
            const c64 den = c64(b0, 0.0) + b1 * zinv + b2 * zinv2;
            const c64 den2 = den * den;
            const c64 den4 = den2 * den2;
            const c64 num1 = c64(b0_1, 0.0) + b1_1 * zinv;
            const c64 num2 = c64(a0, 0.0) + a12 * zinv;
            const c64 num3 = c64(a0, 0.0) + a13 * zinv;
            const c64 num4 = c64(a0, 0.0) + a14 * zinv;
            const c64 h_tot = (num1 * num2 * num3 * num4) / den4;
            row[k] = keel::abs(h_tot) * scale;
        }
    }
    return AudioStatus::kOk;
}

AudioStatus cepstral_coefficients(ArrayRef<f64> spectra_col_major, usize num_bands, usize num_frames,
                                  usize num_coeffs, MutArrayRef<f64> coeffs_col_major,
                                  CepstralRectification rect, Allocator* alloc) {
    if (num_coeffs <= 1) {
        return AudioStatus::kInvalidArgument;
    }
    if (num_bands == 0 || num_frames == 0) {
        return AudioStatus::kOk;
    }
    const usize in_total = num_bands * num_frames;
    const usize out_total = num_frames * num_coeffs;
    if (spectra_col_major.size() < in_total || coeffs_col_major.size() < out_total) {
        return AudioStatus::kInvalidArgument;
    }

    if (rect != CepstralRectification::kNone) {
        for (usize i = 0; i < in_total; ++i) {
            if (!(spectra_col_major[i] >= 0.0)) {
                return AudioStatus::kInvalidArgument;
            }
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> basis(num_coeffs * num_bands, scratch.allocator());
    build_dct2_basis(num_bands, num_coeffs, basis);

    const f64 log_floor = keel::log10(Limits<f64>::kMinPositive);
    Array<f64> rectified(in_total, scratch.allocator());
    for (usize i = 0; i < in_total; ++i) {
        rectified[i] = rectify_value(spectra_col_major[i], rect, log_floor);
    }

    for (usize a = 0; a < num_frames; ++a) {
        const usize s_base = a * num_bands;
        for (usize k = 0; k < num_coeffs; ++k) {
            f64 acc = 0.0;
            for (usize l = 0; l < num_bands; ++l) {
                acc += basis[k + l * num_coeffs] * rectified[s_base + l];
            }
            coeffs_col_major[a + k * num_frames] = acc;
        }
    }
    return AudioStatus::kOk;
}

AudioStatus cepstral_coefficients_f32(ArrayRef<f32> spectra_col_major, usize num_bands, usize num_frames,
                                      usize num_coeffs, MutArrayRef<f32> coeffs_col_major,
                                      CepstralRectification rect, Allocator* alloc) {
    if (num_coeffs <= 1) {
        return AudioStatus::kInvalidArgument;
    }
    if (num_bands == 0 || num_frames == 0) {
        return AudioStatus::kOk;
    }
    const usize in_total = num_bands * num_frames;
    const usize out_total = num_frames * num_coeffs;
    if (spectra_col_major.size() < in_total || coeffs_col_major.size() < out_total) {
        return AudioStatus::kInvalidArgument;
    }

    if (rect != CepstralRectification::kNone) {
        for (usize i = 0; i < in_total; ++i) {
            if (!(static_cast<f64>(spectra_col_major[i]) >= 0.0)) {
                return AudioStatus::kInvalidArgument;
            }
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> basis(num_coeffs * num_bands, scratch.allocator());
    build_dct2_basis(num_bands, num_coeffs, basis);

    const f64 log_floor = keel::log10(Limits<f64>::kMinPositive);
    Array<f32> rectified(in_total, scratch.allocator());
    for (usize i = 0; i < in_total; ++i) {
        rectified[i] = static_cast<f32>(rectify_value(static_cast<f64>(spectra_col_major[i]), rect, log_floor));
    }

    for (usize a = 0; a < num_frames; ++a) {
        const usize s_base = a * num_bands;
        for (usize k = 0; k < num_coeffs; ++k) {
            f32 acc = 0.0f;
            for (usize l = 0; l < num_bands; ++l) {
                acc += static_cast<f32>(basis[k + l * num_coeffs]) * rectified[s_base + l];
            }
            coeffs_col_major[a + k * num_frames] = acc;
        }
    }
    return AudioStatus::kOk;
}

AudioStatus mfcc(ArrayRef<f64> x, f64 sample_rate, usize num_coeffs, MutArrayRef<f64> coeffs_col_major,
                 MutArrayRef<f64> delta_col_major, MutArrayRef<f64> delta_delta_col_major, Allocator* alloc) {
    if (!(sample_rate > 0.0) || !keel::is_finite(sample_rate) || num_coeffs <= 1) {
        return AudioStatus::kInvalidArgument;
    }

    const FrameGeometry fg = compute_frame_geometry(x.size(), sample_rate, 0.03, 0.02);
    if (fg.num_frames == 0 || fg.win_len == 0) {
        return AudioStatus::kOk;
    }
    const usize num_cols = num_coeffs + 1;
    const usize out_total = fg.num_frames * num_cols;
    if (coeffs_col_major.size() < out_total) {
        return AudioStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize fft_len = fg.win_len;
    const usize h = fft_len / 2 + 1;
    constexpr usize kNumEdges = 42;
    constexpr usize kNumBands = kNumEdges - 2;

    Array<f64> win(fg.win_len, scratch.allocator());
    hamming_periodic(win);

    Array<f64> edges(kNumEdges, scratch.allocator());
    slaney_mel_band_edges(edges);

    Array<f64> fb(h * kNumBands, scratch.allocator());
    const AudioStatus fb_st = slaney_mel_filterbank(sample_rate, fft_len, edges, fb);
    if (fb_st != AudioStatus::kOk) {
        return fb_st;
    }

    auto plan_res = fft::FftPlan1D::create(fft_len, scratch.allocator());
    if (!plan_res.is_ok()) {
        return AudioStatus::kInvalidArgument;
    }
    const fft::FftPlan1D& plan = plan_res.value();

    Array<f64> frame(fg.win_len, scratch.allocator());
    Array<c64> half_spec(h, scratch.allocator());
    Array<f64> mag(h, scratch.allocator());
    Array<f64> mel_mag(kNumBands * fg.num_frames, scratch.allocator());
    Array<f64> log_e(fg.num_frames, scratch.allocator());

    for (usize f = 0; f < fg.num_frames; ++f) {
        const usize start = f * fg.hop;
        f64 energy = 0.0;
        for (usize i = 0; i < fg.win_len; ++i) {
            const f64 xi = x[start + i];
            energy += xi * xi;
            frame[i] = xi * win[i];
        }
        if (energy == 0.0) {
            energy = Limits<f64>::kMinPositive;
        }
        log_e[f] = keel::log(energy);

        plan.exec_rfft(frame, half_spec);
        for (usize j = 0; j < h; ++j) {
            mag[j] = keel::abs(half_spec[j]);
        }

        for (usize b = 0; b < kNumBands; ++b) {
            f64 s = 0.0;
            const f64* fb_col = fb.data() + b * h;
            for (usize j = 0; j < h; ++j) {
                s += fb_col[j] * mag[j];
            }
            mel_mag[b + f * kNumBands] = s;
        }
    }

    Array<f64> cep(fg.num_frames * num_coeffs, scratch.allocator());
    const AudioStatus cep_st =
        cepstral_coefficients(mel_mag, kNumBands, fg.num_frames, num_coeffs, cep, CepstralRectification::kLog,
                              scratch.allocator());
    if (cep_st != AudioStatus::kOk) {
        return cep_st;
    }

    for (usize f = 0; f < fg.num_frames; ++f) {
        coeffs_col_major[f] = log_e[f];
    }
    for (usize n = 0; n < num_coeffs; ++n) {
        for (usize f = 0; f < fg.num_frames; ++f) {
            coeffs_col_major[f + (n + 1) * fg.num_frames] = cep[f + n * fg.num_frames];
        }
    }

    return compute_deltas(ArrayRef<f64>(coeffs_col_major.data(), out_total), fg.num_frames, num_cols,
                          delta_col_major, delta_delta_col_major, scratch.allocator());
}

AudioStatus gtcc(ArrayRef<f64> x, f64 sample_rate, usize num_coeffs, MutArrayRef<f64> coeffs_col_major,
                 MutArrayRef<f64> delta_col_major, MutArrayRef<f64> delta_delta_col_major, Allocator* alloc) {
    if (!(sample_rate > 0.0) || !keel::is_finite(sample_rate) || num_coeffs <= 1) {
        return AudioStatus::kInvalidArgument;
    }

    const FrameGeometry fg = compute_frame_geometry(x.size(), sample_rate, 0.03, 0.02);
    const f64 f_low = 50.0;
    const f64 f_high = sample_rate * 0.5;
    const f64 low_erb = hz_to_erb(f_low);
    const f64 high_erb = hz_to_erb(f_high);
    const f64 diff_erb = keel::ceil(high_erb - low_erb);
    const usize num_bands = (diff_erb > 0.0) ? static_cast<usize>(diff_erb) : 0;

    if (fg.num_frames == 0 || fg.win_len == 0 || num_bands == 0) {
        return AudioStatus::kOk;
    }
    const usize num_cols = num_coeffs + 1;
    const usize out_total = fg.num_frames * num_cols;
    if (coeffs_col_major.size() < out_total) {
        return AudioStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize fft_len = fg.win_len;
    const usize h = fft_len / 2 + 1;

    Array<f64> win(fg.win_len, scratch.allocator());
    hamming_periodic(win);

    Array<f64> fc(num_bands, scratch.allocator());
    if (num_bands == 1) {
        fc[0] = erb_to_hz(low_erb);
    } else {
        const f64 step = (high_erb - low_erb) / static_cast<f64>(num_bands - 1);
        for (usize i = 0; i < num_bands; ++i) {
            fc[i] = erb_to_hz(low_erb + step * static_cast<f64>(i));
        }
    }

    Array<f64> bank(num_bands * h, scratch.allocator());
    const AudioStatus gb_st = gammatone_filterbank(sample_rate, fc, fft_len, bank, true);
    if (gb_st != AudioStatus::kOk) {
        return gb_st;
    }

    auto plan_res = fft::FftPlan1D::create(fft_len, scratch.allocator());
    if (!plan_res.is_ok()) {
        return AudioStatus::kInvalidArgument;
    }
    const fft::FftPlan1D& plan = plan_res.value();

    Array<f64> frame(fg.win_len, scratch.allocator());
    Array<c64> half_spec(h, scratch.allocator());
    Array<f64> mag(h, scratch.allocator());
    Array<f64> band_mag(num_bands * fg.num_frames, scratch.allocator());
    Array<f64> log_e(fg.num_frames, scratch.allocator());

    const bool nyquist_single = ((fft_len & 1u) == 0u);
    for (usize f = 0; f < fg.num_frames; ++f) {
        const usize start = f * fg.hop;
        f64 energy = 0.0;
        for (usize i = 0; i < fg.win_len; ++i) {
            const f64 xi = x[start + i];
            energy += xi * xi;
            frame[i] = xi * win[i];
        }
        if (energy == 0.0) {
            energy = Limits<f64>::kMinPositive;
        }
        log_e[f] = keel::log(energy);

        plan.exec_rfft(frame, half_spec);
        for (usize j = 0; j < h; ++j) {
            mag[j] = keel::abs(half_spec[j]);
        }

        for (usize b = 0; b < num_bands; ++b) {
            const f64* bank_row = bank.data() + b * h;
            f64 s = bank_row[0] * mag[0];
            for (usize k = 1; k + 1 < h; ++k) {
                s += 2.0 * bank_row[k] * mag[k];
            }
            if (h > 1) {
                const f64 last = bank_row[h - 1] * mag[h - 1];
                s += nyquist_single ? last : (2.0 * last);
            }
            band_mag[b + f * num_bands] = s;
        }
    }

    Array<f64> cep(fg.num_frames * num_coeffs, scratch.allocator());
    const AudioStatus cep_st =
        cepstral_coefficients(band_mag, num_bands, fg.num_frames, num_coeffs, cep, CepstralRectification::kLog,
                              scratch.allocator());
    if (cep_st != AudioStatus::kOk) {
        return cep_st;
    }

    for (usize f = 0; f < fg.num_frames; ++f) {
        coeffs_col_major[f] = log_e[f];
    }
    for (usize n = 0; n < num_coeffs; ++n) {
        for (usize f = 0; f < fg.num_frames; ++f) {
            coeffs_col_major[f + (n + 1) * fg.num_frames] = cep[f + n * fg.num_frames];
        }
    }

    return compute_deltas(ArrayRef<f64>(coeffs_col_major.data(), out_total), fg.num_frames, num_cols,
                          delta_col_major, delta_delta_col_major, scratch.allocator());
}

} // namespace keel::audio
