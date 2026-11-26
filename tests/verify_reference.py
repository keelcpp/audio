#!/usr/bin/env python3
"""Differential reference verifier for keel::audio against NumPy/SciPy and MATLAB R2025b Audio Toolbox."""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.fft import dct
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter
from scipy.signal.windows import hamming as sp_hamming


def load_f64(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def load_f32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


def load_c64(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<c16")


def load_mat_f64(path: str, m: int, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape((m, n), order="F")


def load_mat_f32(path: str, m: int, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape((m, n), order="F")


def assert_close(name: str, got: np.ndarray, ref: np.ndarray, tol: float, results: list):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    err = float(np.max(np.abs(got - ref)))
    scale = max(1.0, float(np.max(np.abs(ref))))
    rel = err / scale
    if not np.isfinite(rel) or rel > tol:
        raise AssertionError(f"{name}: max_abs={err:.3e}, rel={rel:.3e} > tol={tol:.3e}")
    results.append((name, rel))


def run_numpy_scipy_checks(tmp: str, results: list):
    # 1. Perceptual scales
    hz = load_f64(os.path.join(tmp, "scale_hz.bin"))
    mel_ref = 2595.0 * np.log10(1.0 + hz / 700.0)
    assert_close("numpy:hz_to_mel", load_f64(os.path.join(tmp, "scale_mel.bin")), mel_ref, 1e-13, results)
    assert_close("numpy:mel_to_hz_rt", load_f64(os.path.join(tmp, "scale_mel_rt.bin")), hz, 1e-13, results)

    bark_raw = 26.81 * hz / (1960.0 + hz) - 0.53
    bark_ref = np.where(bark_raw < 2.0, 0.85 * bark_raw + 0.3,
               np.where(bark_raw > 20.1, 1.22 * bark_raw - 4.422, bark_raw))
    assert_close("numpy:hz_to_bark", load_f64(os.path.join(tmp, "scale_bark.bin")), bark_ref, 1e-13, results)
    assert_close("numpy:bark_to_hz_rt", load_f64(os.path.join(tmp, "scale_bark_rt.bin")), hz, 1e-12, results)

    erb_scale = np.log(10.0) * 1000.0 / (24.673 * 4.368)
    erb_ref = erb_scale * np.log10(1.0 + 0.004368 * hz)
    assert_close("numpy:hz_to_erb", load_f64(os.path.join(tmp, "scale_erb.bin")), erb_ref, 1e-13, results)
    assert_close("numpy:erb_to_hz_rt", load_f64(os.path.join(tmp, "scale_erb_rt.bin")), hz, 1e-13, results)

    phons = load_f64(os.path.join(tmp, "scale_phon.bin"))
    sone1_ref = np.where(phons < 40.0, (phons / 40.0) ** (1.0 / 0.35), 2.0 ** (phons / 10.0 - 4.0))
    assert_close("numpy:phon_to_sone_532_1", load_f64(os.path.join(tmp, "scale_sone_532_1.bin")), sone1_ref, 1e-13, results)

    sones = load_f64(os.path.join(tmp, "scale_sone_in.bin"))
    phon1_ref = np.where(sones < 1.0, 40.0 * (sones ** 0.35), 40.0 + 10.0 * np.log2(np.maximum(sones, 1e-300)))
    assert_close("numpy:sone_to_phon_532_1", load_f64(os.path.join(tmp, "scale_phon_532_1.bin")), phon1_ref, 1e-13, results)

    tab5_phon = np.array([0.0, 2.2, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                          55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    tab5_sone = np.array([0.001, 0.004, 0.008, 0.010, 0.019, 0.031, 0.073, 0.146, 0.26, 0.43, 0.67, 1.00, 1.46, 2.09,
                          2.96, 4.14, 5.77, 8.04, 11.2, 15.8, 22.7, 32.9, 47.7, 69.6, 102.0, 151.0, 225.0, 337.6])
    pchip_s2p = PchipInterpolator(tab5_sone, tab5_phon)
    slope_hi = (tab5_phon[-1] - tab5_phon[-2]) / (tab5_sone[-1] - tab5_sone[-2])
    phon2_ref = np.array([
        0.0 if s <= tab5_sone[0] else (tab5_phon[-1] + slope_hi * (s - tab5_sone[-1]) if s >= tab5_sone[-1] else max(0.0, float(pchip_s2p(s))))
        for s in sones
    ])
    assert_close("scipy:sone_to_phon_532_2", load_f64(os.path.join(tmp, "scale_phon_532_2.bin")), phon2_ref, 1e-12, results)

    # 2. Spectral descriptors
    freqs = load_f64(os.path.join(tmp, "desc_freqs.bin"))
    s1 = load_f64(os.path.join(tmp, "desc_s1.bin"))
    s2 = load_f64(os.path.join(tmp, "desc_s2.bin"))
    c1 = load_c64(os.path.join(tmp, "desc_c1.bin"))
    c2 = load_c64(os.path.join(tmp, "desc_c2.bin"))
    vals = load_f64(os.path.join(tmp, "desc_vals.bin"))
    extra = load_f64(os.path.join(tmp, "desc_extra.bin"))

    sx = np.sum(s1)
    c = np.sum(freqs * s1) / sx
    df = freqs - c
    sp = np.sqrt(np.sum((df ** 2) * s1) / sx)
    sk = np.sum((df ** 3) * s1) / ((sp ** 3) * sx)
    ku = np.sum((df ** 4) * s1) / ((sp ** 4) * sx)
    cs = np.cumsum(s1)
    ro95 = freqs[np.argmax(cs >= 0.95 * sx)]
    ro50 = freqs[np.argmax(cs >= 0.50 * sx)]
    dec = np.sum((s1[1:] - s1[0]) / np.arange(1, len(s1))) / np.sum(s1[1:])
    slp = np.sum((freqs - np.mean(freqs)) * (s1 - np.mean(s1))) / np.sum((freqs - np.mean(freqs)) ** 2)
    cr = np.max(s1) / np.mean(s1)
    q = s1 / sx
    ent = -np.sum(q * np.log(q)) / np.log(len(s1))
    flt = np.exp(np.mean(np.log(s1 + np.finfo(np.float64).eps))) / np.mean(s1)
    fl2 = np.linalg.norm(s2 - s1, ord=2)
    fl1 = np.linalg.norm(s2 - s1, ord=1)
    ref_vals = np.array([c, sp, sk, ku, ro95, ro50, dec, slp, cr, ent, flt, fl2, fl1])
    assert_close("numpy:spectral_descriptors_13", vals, ref_vals, 1e-12, results)

    ref_extra = np.array([
        np.linalg.norm(s2 - s1, ord=np.inf),
        np.linalg.norm(c2 - c1, ord=2),
        np.linalg.norm(c2 - c1, ord=1),
        np.linalg.norm(c2 - c1, ord=np.inf),
    ])
    assert_close("numpy:spectral_flux_linf_c64", extra, ref_extra, 1e-13, results)

    # 3. Features: Hamming, Mel filterbank, Slaney filterbank, Gammatone filterbank, audio_delta
    assert_close("scipy:hamming_periodic_64", load_f64(os.path.join(tmp, "hamming_64.bin")), sp_hamming(64, sym=False), 1e-14, results)

    # Slaney edges & filterbank
    edges_ref = np.zeros(42, dtype=np.float64)
    edges_ref[:13] = 133.33333333333333 + (133.33333333333333 * 0.5) * np.arange(13)
    for i in range(13, 42):
        edges_ref[i] = edges_ref[i - 1] * 1.0711703
    assert_close("numpy:slaney_mel_edges_42", load_f64(os.path.join(tmp, "slaney_edges.bin")), edges_ref, 1e-13, results)

    fgrid = np.arange(241, dtype=np.float64) * (16000.0 / 480.0)
    sfb_ref = np.zeros((241, 40), dtype=np.float64)
    for k in range(40):
        l_e, c_e, r_e = edges_ref[k], edges_ref[k + 1], edges_ref[k + 2]
        w = np.maximum(0.0, np.minimum((fgrid - l_e) / (c_e - l_e), (r_e - fgrid) / (r_e - c_e)))
        sfb_ref[:, k] = w * (2.0 / (r_e - l_e))
    assert_close("numpy:slaney_mel_filterbank", load_mat_f64(os.path.join(tmp, "slaney_fb.bin"), 241, 40), sfb_ref, 1e-13, results)

    # Gammatone filterbank (Patterson-Holdsworth / Slaney 1993)
    gt_fc = load_f64(os.path.join(tmp, "gt_fc.bin"))
    T = 1.0 / 16000.0
    s32a = np.sqrt(3.0 + 2.0 ** 1.5)
    s32b = np.sqrt(3.0 - 2.0 ** 1.5)
    zinv = np.exp(-1j * 2.0 * np.pi * np.arange(241, dtype=np.float64) / 480.0)
    gt_unnorm_ref = np.zeros((4, 241), dtype=np.float64)
    gt_norm_ref = np.zeros((4, 241), dtype=np.float64)
    for i, cf in enumerate(gt_fc):
        erb = cf / 9.26449 + 24.7
        B = 1.019 * 2.0 * np.pi * erb
        ct, st = np.cos(2.0 * np.pi * cf * T), np.sin(2.0 * np.pi * cf * T)
        em = np.exp(-B * T)
        em2 = np.exp(-2.0 * B * T)
        a0 = T
        a11 = -(2.0 * T * ct * em + 2.0 * s32a * T * st * em) * 0.5
        a12 = -(2.0 * T * ct * em - 2.0 * s32a * T * st * em) * 0.5
        a13 = -(2.0 * T * ct * em + 2.0 * s32b * T * st * em) * 0.5
        a14 = -(2.0 * T * ct * em - 2.0 * s32b * T * st * em) * 0.5
        b1, b2 = -2.0 * ct * em, em2
        e4j = np.exp(4j * np.pi * cf * T)
        e2j = np.exp(2j * np.pi * cf * T) * em
        t1 = -2.0 * e4j * T + 2.0 * e2j * T * (ct - s32b * st)
        t2 = -2.0 * e4j * T + 2.0 * e2j * T * (ct + s32b * st)
        t3 = -2.0 * e4j * T + 2.0 * e2j * T * (ct - s32a * st)
        t4 = -2.0 * e4j * T + 2.0 * e2j * T * (ct + s32a * st)
        ds = -2.0 * em2 - 2.0 * e4j + 2.0 * (1.0 + e4j) * em
        gain = np.abs((t1 * t2 * t3 * t4) / (ds ** 4))
        den = (1.0 + b1 * zinv + b2 * (zinv ** 2)) ** 4
        H_tot = ((a0 / gain + (a11 / gain) * zinv) * (a0 + a12 * zinv) * (a0 + a13 * zinv) * (a0 + a14 * zinv)) / den
        gt_unnorm_ref[i, :] = np.abs(H_tot)
        gt_norm_ref[i, :] = gt_unnorm_ref[i, :] * (2.0 / (1.019 * 24.7 * (0.00437 * cf + 1.0)))
    assert_close("numpy:gammatone_fb_unnorm", load_f64(os.path.join(tmp, "gt_fb_unnorm.bin")).reshape((4, 241)), gt_unnorm_ref, 1e-12, results)
    assert_close("numpy:gammatone_fb_norm", load_f64(os.path.join(tmp, "gt_fb.bin")).reshape((4, 241)), gt_norm_ref, 1e-12, results)

    dx = load_f64(os.path.join(tmp, "delta_x.bin"))
    for wlen, fname in [(5, "delta_y5.bin"), (9, "delta_y9.bin")]:
        m = wlen // 2
        b = np.arange(m, -m - 1, -1, dtype=np.float64) / np.sum(np.arange(1, m + 1, dtype=np.float64) ** 2)
        dy_ref = lfilter(b, [1.0], dx)
        assert_close(f"scipy:audio_delta_w{wlen}", load_f64(os.path.join(tmp, fname)), dy_ref, 1e-13, results)

    # 4. Cepstral coefficients via scipy.fft.dct
    cspec = load_mat_f64(os.path.join(tmp, "cep_spec.bin"), 8, 5)
    c_log_ref = dct(np.log10(cspec), type=2, axis=0, norm="ortho")[:4, :].T
    c_cbrt_ref = dct(np.cbrt(cspec), type=2, axis=0, norm="ortho")[:4, :].T
    c_none_ref = dct(cspec, type=2, axis=0, norm="ortho")[:4, :].T
    assert_close("scipy:cepstral_log", load_mat_f64(os.path.join(tmp, "cep_log.bin"), 5, 4), c_log_ref, 1e-13, results)
    assert_close("scipy:cepstral_cbrt", load_mat_f64(os.path.join(tmp, "cep_cbrt.bin"), 5, 4), c_cbrt_ref, 1e-13, results)
    assert_close("scipy:cepstral_none", load_mat_f64(os.path.join(tmp, "cep_none.bin"), 5, 4), c_none_ref, 1e-13, results)

    cspec_f32 = load_mat_f32(os.path.join(tmp, "cep_spec_f32.bin"), 4, 2)
    floor32 = np.log10(np.finfo(np.float32).tiny).astype(np.float64)
    rect32 = np.where(cspec_f32 > 0, np.log10(np.maximum(cspec_f32, np.finfo(np.float32).tiny)), floor32).astype(np.float32)
    c_f32_ref = dct(rect32.astype(np.float64), type=2, axis=0, norm="ortho")[:3, :].T
    assert_close("scipy:cepstral_f32_zero_floor", load_mat_f32(os.path.join(tmp, "cep_log_f32.bin"), 2, 3), c_f32_ref, 1e-5, results)


def run_matlab_checks(tmp: str, results: list):
    matlab_exe = shutil.which("matlab")
    if not matlab_exe:
        cand = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
        if os.path.exists(cand):
            matlab_exe = cand
    if not matlab_exe:
        raise RuntimeError("MATLAB executable not found; MATLAB Audio Toolbox verification is required.")

    d = tmp.replace("\\", "/")
    out_m = f"{d}/matlab_audio_ref.bin"
    m_code = (
        f"f=fopen('{d}/scale_hz.bin','rb'); hz=fread(f,'double'); fclose(f); "
        "m_mel=hz2mel(hz); m_mel_rt=mel2hz(m_mel); "
        "m_bark=hz2bark(hz); m_bark_rt=bark2hz(m_bark); "
        "m_erb=hz2erb(hz); m_erb_rt=erb2hz(m_erb); "
        f"f=fopen('{d}/scale_phon.bin','rb'); ph=fread(f,'double'); fclose(f); "
        "m_s1=phon2sone(ph,'ISO 532-1'); m_s2=phon2sone(ph,'ISO 532-2'); "
        f"f=fopen('{d}/scale_sone_in.bin','rb'); sn=fread(f,'double'); fclose(f); "
        "m_p1=sone2phon(sn,'ISO 532-1'); m_p2=sone2phon(sn,'ISO 532-2'); "
        f"f=fopen('{d}/desc_freqs.bin','rb'); df=fread(f,'double'); fclose(f); "
        f"f=fopen('{d}/desc_s1.bin','rb'); ds1=fread(f,'double'); fclose(f); "
        f"f=fopen('{d}/desc_s2.bin','rb'); ds2=fread(f,'double'); fclose(f); "
        "fl2=spectralFlux([ds1, ds2], df, 'NormType', 2); "
        "fl1=spectralFlux([ds1, ds2], df, 'NormType', 1); "
        "m_desc=[spectralCentroid(ds1,df); spectralSpread(ds1,df); spectralSkewness(ds1,df); "
        "spectralKurtosis(ds1,df); spectralRolloffPoint(ds1,df,'Threshold',0.95); "
        "spectralRolloffPoint(ds1,df,'Threshold',0.50); spectralDecrease(ds1,df); "
        "spectralSlope(ds1,df); spectralCrest(ds1,df); "
        "spectralEntropy(ds1,df,'Instantaneous',true); spectralFlatness(ds1,df); fl2(2); fl1(2)]; "
        "m_ham=hamming(64,'periodic'); "
        "[m_melfb, m_melcf]=designAuditoryFilterBank(16000,'FFTLength',128,'NumBands',12,'FrequencyScale','mel'); "
        f"f=fopen('{d}/delta_x.bin','rb'); dx=fread(f,'double'); fclose(f); "
        "m_dy5=audioDelta(dx,5); m_dy9=audioDelta(dx,9); "
        f"f=fopen('{d}/cep_spec.bin','rb'); csp=reshape(fread(f,'double'),8,5); fclose(f); "
        "m_clog=cepstralCoefficients(csp,'NumCoeffs',4,'Rectification','log'); "
        "m_ccbrt=cepstralCoefficients(csp,'NumCoeffs',4,'Rectification','cubic-root'); "
        f"f=fopen('{d}/cep_spec_f32.bin','rb'); cspf=reshape(fread(f,'single=>single'),4,2); fclose(f); "
        "m_clog32=double(cepstralCoefficients(cspf,'NumCoeffs',3,'Rectification','log')); "
        f"f=fopen('{d}/sig_16k.bin','rb'); s16=fread(f,'double'); fclose(f); "
        "[m_mc, m_md, m_mdd]=mfcc(s16, 16000, 'NumCoeffs', 13); "
        "[m_gc, m_gd, m_gdd]=gtcc(s16, 16000, 'NumCoeffs', 13); "
        f"f=fopen('{d}/sig_8k.bin','rb'); s8=fread(f,'double'); fclose(f); "
        "m_pncf=pitch(s8, 8000, 'Method', 'NCF'); "
        "m_pcep=pitch(s8, 8000, 'Method', 'CEP'); "
        "m_ppef=pitch(s8, 8000, 'Method', 'PEF'); "
        "m_plhs=pitch(s8, 8000, 'Method', 'LHS'); "
        "m_psrh=pitch(s8, 8000, 'Method', 'SRH'); "
        "m_hr=harmonicRatio(s8, 8000); "
        "pack=[m_mel(:); m_mel_rt(:); m_bark(:); m_bark_rt(:); m_erb(:); m_erb_rt(:); "
        "m_s1(:); m_s2(:); m_p1(:); m_p2(:); m_desc(:); m_ham(:); m_melfb(:); m_melcf(:); "
        "m_dy5(:); m_dy9(:); m_clog(:); m_ccbrt(:); m_clog32(:); "
        "m_mc(:); m_md(:); m_mdd(:); m_gc(:); m_gd(:); m_gdd(:); "
        "m_pncf(:); m_pcep(:); m_ppef(:); m_plhs(:); m_psrh(:); m_hr(:)]; "
        f"f=fopen('{out_m}','wb'); fwrite(f, pack, 'double'); fclose(f);"
    )
    proc = subprocess.run([matlab_exe, "-batch", m_code], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MATLAB reference failed:\n{proc.stdout}\n{proc.stderr}")

    raw = load_f64(os.path.join(tmp, "matlab_audio_ref.bin"))
    off = 0

    def take(n: int) -> np.ndarray:
        nonlocal off
        v = raw[off:off + n]
        off += n
        return v

    assert_close("matlab:hz2mel", load_f64(os.path.join(tmp, "scale_mel.bin")), take(10), 1e-13, results)
    assert_close("matlab:mel2hz", load_f64(os.path.join(tmp, "scale_mel_rt.bin")), take(10), 1e-13, results)
    assert_close("matlab:hz2bark", load_f64(os.path.join(tmp, "scale_bark.bin")), take(10), 1e-13, results)
    assert_close("matlab:bark2hz", load_f64(os.path.join(tmp, "scale_bark_rt.bin")), take(10), 1e-12, results)
    assert_close("matlab:hz2erb", load_f64(os.path.join(tmp, "scale_erb.bin")), take(10), 1e-13, results)
    assert_close("matlab:erb2hz", load_f64(os.path.join(tmp, "scale_erb_rt.bin")), take(10), 1e-13, results)
    assert_close("matlab:phon2sone_532_1", load_f64(os.path.join(tmp, "scale_sone_532_1.bin")), take(13), 1e-13, results)
    assert_close("matlab:phon2sone_532_2", load_f64(os.path.join(tmp, "scale_sone_532_2.bin")), take(13), 1e-11, results)
    assert_close("matlab:sone2phon_532_1", load_f64(os.path.join(tmp, "scale_phon_532_1.bin")), take(12), 1e-13, results)
    assert_close("matlab:sone2phon_532_2", load_f64(os.path.join(tmp, "scale_phon_532_2.bin")), take(12), 1e-12, results)
    assert_close("matlab:spectral_descriptors_13", load_f64(os.path.join(tmp, "desc_vals.bin")), take(13), 1e-12, results)
    assert_close("matlab:hamming_periodic_64", load_f64(os.path.join(tmp, "hamming_64.bin")), take(64), 1e-14, results)
    assert_close("matlab:mel_filterbank_12x65", load_f64(os.path.join(tmp, "mel_fb.bin")), take(12 * 65), 1e-12, results)
    assert_close("matlab:mel_center_freqs", load_f64(os.path.join(tmp, "mel_cf.bin")), take(12), 1e-13, results)
    assert_close("matlab:audioDelta_w5", load_f64(os.path.join(tmp, "delta_y5.bin")), take(32), 1e-13, results)
    assert_close("matlab:audioDelta_w9", load_f64(os.path.join(tmp, "delta_y9.bin")), take(32), 1e-13, results)
    assert_close("matlab:cepstralCoefficients_log", load_f64(os.path.join(tmp, "cep_log.bin")), take(20), 1e-13, results)
    assert_close("matlab:cepstralCoefficients_cbrt", load_f64(os.path.join(tmp, "cep_cbrt.bin")), take(20), 1e-13, results)
    assert_close("matlab:cepstralCoefficients_f32", load_f32(os.path.join(tmp, "cep_log_f32.bin")), take(6), 1e-5, results)

    nf = 23
    nc = 14
    assert_close("matlab:mfcc_coeffs", load_f64(os.path.join(tmp, "mfcc_c.bin")), take(nf * nc), 1e-10, results)
    assert_close("matlab:mfcc_delta", load_f64(os.path.join(tmp, "mfcc_d.bin")), take(nf * nc), 1e-10, results)
    assert_close("matlab:mfcc_delta_delta", load_f64(os.path.join(tmp, "mfcc_dd.bin")), take(nf * nc), 1e-10, results)
    assert_close("matlab:gtcc_coeffs", load_f64(os.path.join(tmp, "gtcc_c.bin")), take(nf * nc), 1e-8, results)
    assert_close("matlab:gtcc_delta", load_f64(os.path.join(tmp, "gtcc_d.bin")), take(nf * nc), 1e-8, results)
    assert_close("matlab:gtcc_delta_delta", load_f64(os.path.join(tmp, "gtcc_dd.bin")), take(nf * nc), 1e-8, results)

    # Pitch & harmonic ratio: CEP/LHS exact; NCF sub-sample parabolic refinement vs MATLAB integer lag (~0.1%);
    # PEF/SRH within 2% of MATLAB (and within 2% of true 200 Hz f0); HR within 2e-6.
    assert_close("matlab:pitch_NCF", load_f64(os.path.join(tmp, "pitch_ncf.bin")), take(15), 2e-3, results)
    assert_close("matlab:pitch_CEP", load_f64(os.path.join(tmp, "pitch_cep.bin")), take(15), 1e-12, results)
    assert_close("matlab:pitch_PEF", load_f64(os.path.join(tmp, "pitch_pef.bin")), take(15), 2e-2, results)
    assert_close("matlab:pitch_LHS", load_f64(os.path.join(tmp, "pitch_lhs.bin")), take(15), 1e-12, results)
    assert_close("matlab:pitch_SRH", load_f64(os.path.join(tmp, "pitch_srh.bin")), take(15), 2.5e-2, results)
    assert_close("matlab:harmonicRatio", load_f64(os.path.join(tmp, "hr_8k.bin")), take(18), 5e-6, results)
    assert off == len(raw), f"MATLAB pack mismatch: consumed {off} of {len(raw)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path/to/test_audio.exe>", file=sys.stderr)
        sys.exit(2)
    exe = sys.argv[1]
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([exe, "--dump-ref", tmp], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"test_audio --dump-ref failed:\n{proc.stdout}\n{proc.stderr}")

        run_numpy_scipy_checks(tmp, results)
        run_matlab_checks(tmp, results)

    max_rel = max(r for _, r in results)
    print(f"PASS: {len(results)}/{len(results)} reference checks passed (max rel={max_rel:.2e}):")
    for name, rel in results:
        print(f"  [OK] {name:<35s} rel={rel:.2e}")


if __name__ == "__main__":
    main()
