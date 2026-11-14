"""用 middle 里存档的光谱重做规律迁移，时域只保留能对上光谱的记录。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
MID = Path(r"C:\Work\大四上\大数据处理\middle")
FIG7 = Path(r"C:\Work\杨子悦\陆梁军组实习\Comb文献\jlt2024_repro\fig7_set03_result.mat")
OUT = ROOT / "results" / "middle_transfer_features.csv"
SUMMARY = ROOT / "results" / "middle_transfer_summary.json"

CASES = [
    ("AMX_0", "AMX = 0", {"AMX": 0.0}),
    ("AMX_sign_flip", "AMX 反号", {"amx_sign": -1.0}),
    ("AMX_2p0GHz", "AMX = 2.0 GHz", {"AMX": 2.0e9}),
    ("beta2_anomalous_-600ps2km", "反常色散", {"beta2": -600e-27}),
    ("Ppump_1p20W", "泵浦 1.20 W", {"P_pump": 1.20}),
    ("Ppump_0p01W", "泵浦 0.01 W", {"P_pump": 0.01}),
    ("loss_0p30dBcm", "损耗 0.30 dB/cm", {"loss_dBcm": 0.30}),
    ("p_cross_12", "p_cross = 12", {"p_cross": 12.0}),
    ("detuning_delta_0p12", "失谐 0.12", {"delw": 0.12}),
]


def device() -> dict:
    c = 299792458.0
    radius = 246.0
    fsr_target = 96.2e9
    length = 2 * np.pi * radius * 1e-6
    n_g = c / (fsr_target * length)
    tr = length * n_g / c
    n = 256
    mu = np.arange(-n // 2, n // 2, dtype=float)
    omega = mu * (2 * np.pi / tr)
    return {
        "L": length,
        "FSR": 1.0 / tr,
        "Tr": tr,
        "N": n,
        "mu": mu,
        "omega": omega,
        "NL": 1j * 1.22 * length,
        "base": {
            "P_pump": 0.2,
            "loss_dBcm": 0.04,
            "beta2": 600e-27,
            "seta": 0.011,
            "AMX": 200e6,
            "p_cross": 1.0,
            "amx_sign": 1.0,
        },
    }


def spectrum_of(field: np.ndarray) -> np.ndarray:
    n = field.size
    power = np.abs(np.fft.fftshift(np.fft.fft(field)) / n) ** 2
    return 10 * np.log10(power + np.finfo(float).tiny) + 30.0


def linear_op(dev: dict, params: dict, detune: float) -> np.ndarray:
    n = dev["N"]
    disp = 0.5 * params["beta2"] * dev["omega"] ** 2 * dev["L"]
    i0 = n // 2 + int(params["p_cross"]) - 1
    shift = params["amx_sign"] * 2 * np.pi * params["AMX"] / dev["FSR"]
    if 0 <= i0 < n:
        disp[i0] += shift
    if 0 <= i0 + 1 < n:
        disp[i0 + 1] -= shift
    alpha = 1 - 10 ** (-(params["loss_dBcm"] * 100) * dev["L"] / 10)
    return np.exp(-alpha / 2 - params["seta"] / 2 - 1j * detune + 1j * disp)


def propagate(field: np.ndarray, lin: np.ndarray, nl: complex, drive: float, n_hold: int = 12000, every: int = 200):
    specs, times = [], []
    for step in range(1, n_hold + 1):
        field = field * np.exp(np.abs(field) ** 2 * nl)
        freq = np.fft.fftshift(np.fft.fft(field)) * lin
        field = np.fft.ifft(np.fft.ifftshift(freq)) + drive
        if step % every == 0:
            specs.append(spectrum_of(field))
            times.append(np.abs(field) ** 2)
    return np.asarray(specs), np.asarray(times)


def sech_residual(mu: np.ndarray, power_dbm: np.ndarray) -> float:
    keep = np.abs(mu) > 1
    mu_k = mu[keep]
    linear = 10 ** (power_dbm[keep] / 10)
    if len(mu_k) < 8 or float(np.sum(linear)) <= 0:
        return float("nan")
    center = float(np.sum(linear * mu_k) / np.sum(linear))
    offset = mu_k - center
    best = 1e9
    for width in np.linspace(4, 70, 34):
        envelope = 1 / np.cosh(offset / width) ** 2
        denom = float(np.dot(envelope, envelope))
        if denom <= 0:
            continue
        scale = float(np.dot(envelope, linear) / denom)
        error = np.sqrt(np.mean((linear - scale * envelope) ** 2)) / (float(np.mean(linear)) + 1e-30)
        best = min(best, error)
    return float(best)


def symmetry(mu: np.ndarray, power: np.ndarray) -> tuple[float, float]:
    table = {int(round(mode)): value for mode, value in zip(mu, power)}
    left, right = [], []
    for mode, value in table.items():
        if mode > 0 and -mode in table:
            right.append(value)
            left.append(table[-mode])
    if len(left) < 5:
        return float("nan"), float("nan")
    return float(np.mean(np.abs(np.array(left) - np.array(right)))), float(np.corrcoef(left, right)[0, 1])


def spectral_features(mu: np.ndarray, power: np.ndarray) -> dict:
    peak = float(np.max(power))
    teeth = power > peak - 35
    loud = power > peak - 20
    span = float(mu[loud].max() - mu[loud].min()) if np.any(loud) else 0.0
    mu_t = mu[teeth]
    power_t = power[teeth]
    order = np.argsort(mu_t)
    mu_t, power_t = mu_t[order], power_t[order]
    adjacent = np.abs(np.diff(power_t))
    adjacent = adjacent[np.isclose(np.diff(mu_t), 1)]
    sym_mae, sym_corr = symmetry(mu_t, power_t)
    linear = 10 ** (power_t / 10)
    left = float(linear[mu_t < 0].sum())
    right = float(linear[mu_t > 0].sum())
    even = float(linear[np.round(mu_t) % 2 == 0].sum())
    odd = float(linear[np.round(mu_t) % 2 != 0].sum())
    pump_at = int(np.argmin(np.abs(mu)))
    others = power_t[np.abs(mu_t) > 0]
    envelope = float(np.corrcoef(np.abs(mu_t), power_t)[0, 1]) if np.std(power_t) > 0 else float("nan")
    if span < 50:
        shape = "narrow"
    elif sech_residual(mu_t, power_t) < 0.5:
        shape = "smooth"
    else:
        shape = "modulated"
    return {
        "n_teeth": int(np.sum(teeth)),
        "tooth_span": float(mu[teeth].max() - mu[teeth].min()) if np.count_nonzero(teeth) else 0.0,
        "lines_within_20db": int(np.sum(loud)),
        "mu_span_20db": span,
        "adj_db": float(np.mean(adjacent)) if len(adjacent) else float("nan"),
        "sym_mae": sym_mae,
        "sym_corr": sym_corr,
        "left_share": left / (left + right) if left + right else float("nan"),
        "even_share": even / (even + odd) if even + odd else float("nan"),
        "pump_excess_db": float(power[pump_at] - np.median(others)) if len(others) else float("nan"),
        "neighbor_gap_db": float(power[pump_at] - 0.5 * (power[pump_at - 1] + power[pump_at + 1])),
        "envelope_corr": envelope,
        "sech_rms": sech_residual(mu_t, power_t),
        "shape_rule": shape,
    }


def time_facts(power: np.ndarray) -> dict:
    """环上计数，不随平移改变。一个低区画在中间是坑，把高的一段放中间就是平台。"""
    scale = power / np.max(power)
    low = scale < 0.40
    filled = low.copy()
    gap = 0
    for index, flag in enumerate(low):
        if flag:
            gap = 0
            continue
        gap += 1
        if 0 < gap <= 8:
            filled[index - gap : index + 1] = True
    ring = np.r_[filled[-1], filled]
    n_dip = int(np.sum(np.diff(ring.astype(int)) == 1))
    low_frac = float(np.mean(filled))
    high = scale >= 0.60
    high_cv = float(np.std(scale[high]) / np.mean(scale[high])) if np.any(high) else float("nan")
    if n_dip == 1 and float(np.min(scale)) < 0.25:
        kind = "暗孤子"
        view = "平台" if low_frac > 0.5 else "暗坑"
    elif n_dip >= 2 and low_frac > 0.5:
        kind = "多峰"
        view = "多峰"
    elif n_dip >= 2:
        kind = "多坑"
        view = "多坑"
    else:
        kind = "起伏"
        view = "起伏"
    return {
        "time_kind": kind,
        "time_view": view,
        "pedestal": float(np.mean(scale > 0.60)),
        "low_frac": low_frac,
        "high_cv": high_cv,
        "n_dip": n_dip,
        "t_min": float(np.min(scale)),
        "t_median": float(np.median(scale)),
    }


def main() -> None:
    saved = sio.loadmat(MID / "failed_spectra.mat", squeeze_me=True, struct_as_record=False)
    seed = sio.loadmat(FIG7, squeeze_me=True, struct_as_record=False)
    dev = device()
    origin = int(saved["i0"]) - 1
    amp = dev["N"] * 10 ** ((np.asarray(seed["spec_all"][origin], float) - 30) / 20)
    phase = np.asarray(seed["ph_all"][origin], float)
    field0 = np.fft.ifft(np.fft.ifftshift(amp * np.exp(1j * phase)))
    detune0 = float(saved["delw0"])
    stored = {str(item.name): np.asarray(item.spec_dB, float) for item in saved["rec"]}

    rows = []
    for name, title, patch in CASES:
        params = dict(dev["base"])
        params.update(patch)
        detune = float(params.pop("delw", detune0))
        specs, times = propagate(
            field0.copy(),
            linear_op(dev, params, detune),
            dev["NL"],
            float(np.sqrt(params["seta"] * params["P_pump"])),
        )
        spec = stored[name]
        band = spec > np.max(spec) - 40
        band_err = np.max(np.abs(specs[:, band] - spec[band]), axis=1)
        full_err = np.max(np.abs(specs - spec), axis=1)
        index = int(np.argmin(band_err))
        facts = spectral_features(dev["mu"], spec)
        matched = float(band_err[index]) <= 0.05
        timed = time_facts(times[index]) if matched else {
            "time_kind": "",
            "time_view": "",
            "pedestal": float("nan"),
            "low_frac": float("nan"),
            "high_cv": float("nan"),
            "n_dip": -1,
            "t_min": float("nan"),
            "t_median": float("nan"),
        }
        row = {
            "name": name,
            "title": title,
            "spec_err_db": float(band_err[index]),
            "full_err_db": float(full_err[index]),
            "time_matched": matched,
            **facts,
            **timed,
        }
        rows.append(row)
        print(
            f"{title:16} teeth={row['n_teeth']:3d} span={row['mu_span_20db']:6.1f} "
            f"adj={row['adj_db']:.3f} sym={row['sym_mae']:.3f}/{row['sym_corr']:.3f} "
            f"sech={row['sech_rms']:.3f} pump={row['neighbor_gap_db']:+.2f} "
            f"rule={row['shape_rule']:9} time={row['time_kind'] or '-'} view={row['time_view'] or '-'} "
            f"low={row['low_frac']} dip={row['n_dip']} err={row['spec_err_db']:.3e}"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")
    matched = frame[frame["time_matched"]]
    summary = {
        "n_spectra": int(len(frame)),
        "n_time_matched": int(matched.shape[0]),
        "shape_counts": frame["shape_rule"].value_counts().to_dict(),
        "time_counts": matched["time_kind"].value_counts().to_dict(),
        "view_counts": matched["time_view"].value_counts().to_dict(),
        "dark_rule": matched.loc[matched["time_kind"] == "暗孤子", "shape_rule"].value_counts().to_dict(),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    print("WROTE", OUT)


def draw_library_panel(ax, dev: dict, library, case: int, title: str) -> None:
    row = int(case) - 1
    spec = np.asarray(library["spec"][row], dtype=float)
    time = np.asarray(library["time"][row], dtype=float)
    tau = np.arange(dev["N"]) / dev["N"] * dev["Tr"] * 1e12
    center = int(np.argmin(time) if np.median(time) > 0.75 * np.max(time) else np.argmax(time))
    time = np.roll(time, dev["N"] // 2 - center)
    floor = -45.0
    ax.vlines(dev["mu"], floor, spec, color="#1f4e79", lw=0.7)
    ax.set_xlim(-40, 40)
    visible = spec[np.abs(dev["mu"]) <= 45]
    ax.set_ylim(floor, max(20.0, float(np.max(visible)) + 4))
    ax.set_xlabel(r"模式号 $\mu$")
    ax.set_ylabel("功率 (dBm)", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    right = ax.twinx()
    top = right.twiny()
    top.plot(tau, time, color="#d62728", lw=1.4)
    top.set_xlim(float(tau[0]), float(tau[-1]))
    top.set_xlabel(r"腔内一圈 $\tau$ (ps)", color="#d62728", labelpad=2)
    top.tick_params(axis="x", labelcolor="#d62728", labelsize=8)
    ymax = float(np.max(time)) * 1.08
    right.set_ylim(0, ymax)
    top.set_ylim(0, ymax)
    right.set_ylabel(r"腔内 $|E|^2$", color="#d62728")
    right.tick_params(axis="y", labelcolor="#d62728")
    top.spines["right"].set_color("#d62728")
    top.spines["top"].set_color("#d62728")
    right.spines["right"].set_color("#d62728")
    ax.spines["left"].set_color("#1f4e79")
    ax.spines["bottom"].set_color("#1f4e79")
    ax.set_title(title, fontsize=12, pad=16)
    ax.grid(True, lw=0.3, alpha=0.35)


def plot_panels() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    saved = sio.loadmat(MID / "failed_spectra.mat", squeeze_me=True, struct_as_record=False)
    seed = sio.loadmat(FIG7, squeeze_me=True, struct_as_record=False)
    dev = device()
    origin = int(saved["i0"]) - 1
    amp = dev["N"] * 10 ** ((np.asarray(seed["spec_all"][origin], float) - 30) / 20)
    phase = np.asarray(seed["ph_all"][origin], float)
    field0 = np.fft.ifft(np.fft.ifftshift(amp * np.exp(1j * phase)))
    detune0 = float(saved["delw0"])
    stored = {str(item.name): np.asarray(item.spec_dB, float) for item in saved["rec"]}
    order = [
        ("Ppump_1p20W", "泵浦 1.20 W　时域暗坑"),
        ("beta2_anomalous_-600ps2km", "反常色散　时域起伏"),
        ("AMX_2p0GHz", "AMX = 2.0 GHz　时域多坑"),
    ]
    patches = {name: patch for name, _title, patch in CASES}
    tau = np.arange(dev["N"]) / dev["N"] * dev["Tr"] * 1e12
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    for ax, (name, title) in zip(axes.ravel(), order):
        params = dict(dev["base"])
        params.update(patches[name])
        detune = float(params.pop("delw", detune0))
        _specs, times = propagate(
            field0.copy(),
            linear_op(dev, params, detune),
            dev["NL"],
            float(np.sqrt(params["seta"] * params["P_pump"])),
        )
        spec = stored[name]
        band = spec > np.max(spec) - 40
        errs = np.max(np.abs(_specs[:, band] - spec[band]), axis=1)
        time = times[int(np.argmin(errs))]
        center = int(np.argmin(time) if np.median(time) > 0.75 * np.max(time) else np.argmax(time))
        time = np.roll(time, dev["N"] // 2 - center)
        floor = -45.0
        ax.vlines(dev["mu"], floor, spec, color="#1f4e79", lw=0.7)
        ax.set_xlim(-40, 40)
        ax.set_ylim(floor, max(20.0, float(np.max(spec[np.abs(dev["mu"]) <= 45])) + 4))
        ax.set_xlabel(r"模式号 $\mu$")
        ax.set_ylabel("功率 (dBm)", color="#1f4e79")
        ax.tick_params(axis="y", labelcolor="#1f4e79")
        right = ax.twinx()
        top = right.twiny()
        top.plot(tau, time, color="#d62728", lw=1.4)
        top.set_xlim(float(tau[0]), float(tau[-1]))
        top.set_xlabel(r"腔内一圈 $\tau$ (ps)", color="#d62728", labelpad=2)
        top.tick_params(axis="x", labelcolor="#d62728", labelsize=8)
        ymax = float(np.max(time)) * 1.08
        right.set_ylim(0, ymax)
        top.set_ylim(0, ymax)
        right.set_ylabel(r"腔内 $|E|^2$", color="#d62728")
        right.tick_params(axis="y", labelcolor="#d62728")
        top.spines["right"].set_color("#d62728")
        top.spines["top"].set_color("#d62728")
        right.spines["right"].set_color("#d62728")
        ax.spines["left"].set_color("#1f4e79")
        ax.spines["bottom"].set_color("#1f4e79")
        ax.set_title(title, fontsize=12, pad=16)
        ax.grid(True, lw=0.3, alpha=0.35)
    library = np.load(ROOT / "results" / "dark_library.npz")
    draw_library_panel(axes.ravel()[3], dev, library, case=16, title="扫描 #16　δ=0.060，P=0.40 W")
    fig.tight_layout()
    outs = [
        ROOT / "results" / "user_state_spectra.png",
        MID / "_preview_all_spectra.png",
    ]
    for out in outs:
        fig.savefig(out, dpi=150)
        print("WROTE", out)
    plt.close(fig)
    single, single_ax = plt.subplots(figsize=(6.4, 4.2))
    draw_library_panel(
        single_ax,
        dev,
        library,
        case=15,
        title=r"基准参数　$\delta=0.060$，泵浦 0.20 W，AMX = 200 MHz",
    )
    single.tight_layout()
    single_out = ROOT / "results" / "dark_base_example.png"
    single.savefig(single_out, dpi=160)
    print("WROTE", single_out)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_panels()
    else:
        main()
