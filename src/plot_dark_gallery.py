"""把 58 条停稳光谱和腔内时域画成对照图，供人工指定目标暗孤子。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dark_gallery"


def load_device():
    spec = importlib.util.spec_from_file_location("transfer_middle", ROOT / "src" / "transfer_middle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.device()


def title_of(row: pd.Series) -> str:
    parts = [f"#{int(row['case']):02d}  δ={row['delw']:.3f}  P={row['P_pump']:.2f} W  AMX={row['AMX_MHz']:.0f} MHz"]
    extra = []
    if abs(row["p_cross"] - 1) > 1e-6:
        extra.append(f"p_cross={row['p_cross']:.0f}")
    if abs(row["loss_dBcm"] - 0.04) > 1e-6:
        extra.append(f"损耗={row['loss_dBcm']:.2f}")
    if abs(row["seta"] - 0.011) > 1e-6:
        extra.append(f"θ={row['seta']:.3f}")
    if row["amx_sign"] < 0:
        extra.append("AMX反号")
    if row["beta2_ps2_per_km"] < 0:
        extra.append("反常色散")
    if extra:
        parts.append("  ".join(extra))
    return "\n".join(parts)


def draw_one(ax, mu: np.ndarray, spec: np.ndarray, time: np.ndarray, tau: np.ndarray, title: str) -> None:
    center = int(np.argmin(time) if np.median(time) > 0.75 * np.max(time) else np.argmax(time))
    shown = np.roll(time, len(time) // 2 - center)
    floor = -50.0
    visible = spec[np.abs(mu) <= 50]
    ax.vlines(mu, floor, spec, color="#1f4e79", lw=0.6)
    ax.set_xlim(-50, 50)
    ax.set_ylim(floor, max(15.0, float(np.max(visible)) + 6))
    ax.set_xlabel(r"模式号 $\mu$", fontsize=8, color="#1f4e79")
    ax.set_ylabel("功率 (dBm)", fontsize=8, color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    right = ax.twinx()
    top = right.twiny()
    top.plot(tau, shown, color="#d62728", lw=1.15)
    top.set_xlim(float(tau[0]), float(tau[-1]))
    top.set_xlabel(r"腔内一圈 $\tau$ (ps)", fontsize=8, color="#d62728", labelpad=1)
    top.tick_params(axis="x", labelcolor="#d62728", labelsize=7)
    ymax = float(np.max(shown)) * 1.12
    right.set_ylim(0, ymax)
    top.set_ylim(0, ymax)
    right.set_ylabel(r"$|E|^2$", fontsize=8, color="#d62728")
    right.tick_params(axis="y", labelcolor="#d62728", labelsize=7)
    for spine in ("right", "top"):
        top.spines[spine].set_color("#d62728")
    right.spines["right"].set_color("#d62728")
    ax.spines["left"].set_color("#1f4e79")
    ax.spines["bottom"].set_color("#1f4e79")
    ax.set_title(title, fontsize=9, pad=14, loc="left")
    ax.grid(True, lw=0.25, alpha=0.35)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    dev = load_device()
    table = pd.read_csv(ROOT / "results" / "dark_library.csv")
    pack = np.load(ROOT / "results" / "dark_library.npz")
    mu = pack["mu"]
    tau = np.arange(dev["N"]) / dev["N"] * dev["Tr"] * 1e12
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "all_spectra_time.pdf"
    per_page = 4
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(table), per_page):
            chunk = table.iloc[start : start + per_page]
            fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
            for ax in axes.ravel():
                ax.axis("off")
            for ax, (_, row) in zip(axes.ravel(), chunk.iterrows()):
                ax.axis("on")
                index = int(row["case"]) - 1
                draw_one(ax, mu, pack["spec"][index], pack["time"][index], tau, title_of(row))
            fig.suptitle("蓝柱为光谱，红线为腔内时域。时域已平移到环程中部，只为看清形状。", fontsize=10, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            page = start // per_page + 1
            fig.savefig(OUT / f"page_{page:02d}.png", dpi=140)
            plt.close(fig)
            print("page", page, flush=True)
    print("WROTE", pdf_path)


if __name__ == "__main__":
    main()
