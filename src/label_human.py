"""按人工指定的目标暗孤子编号，给其余记录标形态不完整或混乱。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {1, 8, 15, 22, 23, 28, 29, 30, 35, 36, 37, 38, 49, 50, 51, 52, 55}
# 时域反复振荡，不能看成一个暗坑加平滑背景
CHAOS = {2, 4, 11, 17, 18, 24, 41}
REASON = {
    2: "光谱一侧掉落，时域低侧有一串振荡",
    4: "时域坑边来回起伏",
    11: "泵浦尖峰突出，时域尖峰两侧乱振",
    17: "泵浦齿打断包络，窄坑两侧振荡密",
    18: "泵浦齿打断包络，窄坑两侧振荡密",
    24: "泵浦齿打断包络，窄坑两侧振荡密",
    41: "时域整圈正弦起伏，光谱只剩几根线",
}


def main() -> None:
    table = pd.read_csv(ROOT / "results" / "dark_library.csv")
    labels = []
    for case in table["case"].astype(int):
        if case in TARGETS:
            labels.append("目标暗孤子")
        elif case in CHAOS:
            labels.append("混乱")
        else:
            labels.append("形态不完整")
    table["human_label"] = labels
    table["label_note"] = table["case"].map(lambda case: REASON.get(int(case), ""))
    table.to_csv(ROOT / "results" / "dark_library_labeled.csv", index=False, encoding="utf-8-sig")
    print(table["human_label"].value_counts().to_string())
    print("CHAOS", sorted(CHAOS))
    incomplete = sorted(set(table["case"].astype(int)) - TARGETS - CHAOS)
    print("INCOMPLETE", incomplete)


if __name__ == "__main__":
    main()
