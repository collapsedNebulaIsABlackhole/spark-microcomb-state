"""在正常色散步进上扫参数，保存光谱、腔内时域和时域类别。

标准暗孤子光谱是图 12 第四张对应的存档，即 AMX = 0。
每条参数保存保持结束时的最后一帧，对应实验里停稳后测到的光谱。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
MID = Path(r"C:\Work\大四上\大数据处理\middle")
FIG7 = Path(r"C:\Work\杨子悦\陆梁军组实习\Comb文献\jlt2024_repro\fig7_set03_result.mat")
OUT_NPZ = ROOT / "results" / "dark_library.npz"
OUT_CSV = ROOT / "results" / "dark_library.csv"


def load_stepper():
    spec = importlib.util.spec_from_file_location("transfer_middle", ROOT / "src" / "transfer_middle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_spectrum() -> np.ndarray:
    saved = sio.loadmat(MID / "failed_spectra.mat", squeeze_me=True, struct_as_record=False)
    for item in saved["rec"]:
        if str(item.name) == "AMX_0":
            return np.asarray(item.spec_dB, dtype=float)
    raise RuntimeError("存档中没有 AMX_0")


def shape_corr(mu: np.ndarray, spec: np.ndarray, ref: np.ndarray) -> float:
    wing = (np.abs(mu) >= 2) & (np.abs(mu) <= 45) & (ref > np.max(ref) - 40)
    if np.count_nonzero(wing) < 8:
        return float("nan")
    left = spec[wing]
    right = ref[wing]
    if float(np.std(left)) < 1e-6 or float(np.std(right)) < 1e-6:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def cases(delw0: float) -> list[dict]:
    rows = []
    for delw in (0.03, 0.045, 0.060125, 0.075, 0.09, 0.12):
        for pump in (0.05, 0.10, 0.20, 0.40, 0.80, 1.20):
            rows.append({"delw": delw0 if abs(delw - 0.060125) < 1e-9 else delw, "P_pump": pump})
    for amx in (0.0, 1.0e8, 4.0e8, 8.0e8, 1.2e9, 2.0e9):
        rows.append({"delw": delw0, "P_pump": 0.20, "AMX": amx})
    for pump in (0.10, 0.40, 0.80):
        rows.append({"delw": delw0, "P_pump": pump, "AMX": 0.0})
    for delw in (0.045, 0.075):
        rows.append({"delw": delw, "P_pump": 0.20, "AMX": 0.0})
    for pc in (2.0, 4.0, 8.0, 12.0):
        rows.append({"delw": delw0, "P_pump": 0.20, "p_cross": pc})
    for loss in (0.10, 0.20, 0.30):
        rows.append({"delw": delw0, "P_pump": 0.20, "loss_dBcm": loss})
    for seta in (0.020, 0.050):
        rows.append({"delw": delw0, "P_pump": 0.20, "seta": seta})
    rows.append({"delw": delw0, "P_pump": 0.20, "amx_sign": -1.0})
    rows.append({"delw": delw0, "P_pump": 0.20, "beta2": -600e-27})
    unique = []
    seen = set()
    for row in rows:
        key = (
            round(row["delw"], 6),
            round(row["P_pump"], 4),
            round(row.get("AMX", 200e6), 3),
            round(row.get("p_cross", 1.0), 3),
            round(row.get("loss_dBcm", 0.04), 4),
            round(row.get("beta2", 600e-27), 30),
            round(row.get("seta", 0.011), 4),
            round(row.get("amx_sign", 1.0), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def main() -> None:
    tm = load_stepper()
    dev = tm.device()
    saved = sio.loadmat(MID / "failed_spectra.mat", squeeze_me=True, struct_as_record=False)
    seed = sio.loadmat(FIG7, squeeze_me=True, struct_as_record=False)
    origin = int(saved["i0"]) - 1
    amp = dev["N"] * 10 ** ((np.asarray(seed["spec_all"][origin], float) - 30) / 20)
    phase = np.asarray(seed["ph_all"][origin], float)
    field0 = np.fft.ifft(np.fft.ifftshift(amp * np.exp(1j * phase)))
    delw0 = float(saved["delw0"])
    ref = reference_spectrum()
    plan = cases(delw0)
    print("N_CASES", len(plan), "delw0", delw0, flush=True)

    specs, times, records = [], [], []
    for index, patch in enumerate(plan, start=1):
        params = dict(dev["base"])
        params.update(patch)
        detune = float(params.pop("delw"))
        got_specs, got_times = tm.propagate(
            field0.copy(),
            tm.linear_op(dev, params, detune),
            dev["NL"],
            float(np.sqrt(params["seta"] * params["P_pump"])),
        )
        facts = [tm.time_facts(frame) for frame in got_times]
        dark = [i for i, item in enumerate(facts) if item["time_kind"] == "暗孤子"]
        pick = len(facts) - 1
        chosen = facts[pick]
        spec = got_specs[pick]
        row = {
            "case": index,
            "delw": detune,
            "P_pump": params["P_pump"],
            "AMX_MHz": params["AMX"] / 1e6,
            "p_cross": params["p_cross"],
            "loss_dBcm": params["loss_dBcm"],
            "beta2_ps2_per_km": params["beta2"] / 1e-27,
            "seta": params["seta"],
            "amx_sign": params["amx_sign"],
            "frame": pick,
            "n_dark_frames": len(dark),
            "ref_corr": shape_corr(dev["mu"], spec, ref),
            **{key: chosen[key] for key in ("time_kind", "time_view", "low_frac", "n_dip", "t_min")},
        }
        specs.append(spec)
        times.append(got_times[pick])
        records.append(row)
        print(
            f"{index:02d}/{len(plan)} d={detune:.3f} P={params['P_pump']:.2f} "
            f"AMX={params['AMX']/1e6:.0f} pc={params['p_cross']:.0f} "
            f"loss={params['loss_dBcm']:.2f} kind={chosen['time_kind']} "
            f"dark_frames={len(dark)} corr={row['ref_corr']:.3f}",
            flush=True,
        )

    frame = pd.DataFrame(records)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        OUT_NPZ,
        mu=dev["mu"],
        spec=np.asarray(specs, dtype=np.float32),
        time=np.asarray(times, dtype=np.float32),
        ref=ref.astype(np.float32),
    )
    print(frame["time_kind"].value_counts().to_string())
    print("WROTE", OUT_CSV, OUT_NPZ)


if __name__ == "__main__":
    main()
