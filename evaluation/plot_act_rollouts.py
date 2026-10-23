#!/usr/bin/env python3
"""Plot ACT rollout results from the CSV produced by act_variation_rollout.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_LABELS = {"table": "桌面", "red_above": "红色方块在上", "red_below": "红色方块在下"}
SUCCESS_COLOR = "#228B61"
FAILURE_COLOR = "#C85250"


def load_rollouts(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"episode", "success", "sim_time_s", "cube_target_xy_m", "stack_mode"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: 缺少字段 {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            success = row["success"].strip().lower()
            if success not in {"true", "false"}:
                raise ValueError(f"第 {reader.line_num} 行的 success 应为 True 或 False")
            rows.append({
                "episode": int(row["episode"]),
                "success": success == "true",
                "sim_time_s": float(row["sim_time_s"]),
                "cube_target_xy_m": float(row["cube_target_xy_m"]),
                "stack_mode": row["stack_mode"].strip(),
            })
    if not rows:
        raise ValueError(f"{path}: 没有完成的 rollout")
    return sorted(rows, key=lambda row: row["episode"])


def plot_rollouts(rows: list[dict], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "Noto Sans CJK SC",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    episodes = [row["episode"] for row in rows]
    successes = sum(row["success"] for row in rows)
    colors = [SUCCESS_COLOR if row["success"] else FAILURE_COLOR for row in rows]
    cumulative = []
    count = 0
    for index, row in enumerate(rows, start=1):
        count += row["success"]
        cumulative.append(100 * count / index)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(f"ACT 闭环评估结果  |  {successes}/{len(rows)} 成功 ({successes / len(rows):.1%})", fontsize=17)

    ax = axes[0, 0]
    ax.plot(episodes, cumulative, color="#3366AA", linewidth=2)
    ax.scatter([row["episode"] for row in rows if row["success"]],
               [cumulative[i] for i, row in enumerate(rows) if row["success"]],
               color=SUCCESS_COLOR, zorder=3, label="成功回合")
    ax.set(title="累计成功率", xlabel="回合", ylabel="成功率 (%)", ylim=(0, max(10, max(cumulative) * 1.15)))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    modes = [mode for mode in MODE_LABELS if any(row["stack_mode"] == mode for row in rows)]
    modes.extend(sorted({row["stack_mode"] for row in rows} - set(modes)))
    rates = []
    for mode in modes:
        subset = [row for row in rows if row["stack_mode"] == mode]
        won = sum(row["success"] for row in subset)
        rates.append(100 * won / len(subset))
    bars = ax.bar([MODE_LABELS.get(mode, mode) for mode in modes], rates, color="#6385AF", width=0.6)
    for bar, mode in zip(bars, modes):
        subset = [row for row in rows if row["stack_mode"] == mode]
        won = sum(row["success"] for row in subset)
        ax.annotate(f"{won}/{len(subset)}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center")
    ax.set(title="按初始堆叠方式统计", ylabel="成功率 (%)", ylim=(0, max(15, max(rates) * 1.3)))
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(episodes, [100 * row["cube_target_xy_m"] for row in rows], c=colors, s=35)
    ax.axhline(5, color="#444444", linestyle="--", linewidth=1, label="目标中心 5 cm 阈值")
    ax.set(title="结束时方块距目标中心的水平距离", xlabel="回合", ylabel="距离 (cm)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.bar(episodes, [row["sim_time_s"] for row in rows], color=colors, width=0.85)
    ax.set(title="每回合仿真时长", xlabel="回合", ylabel="秒")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.98, 0.97, "绿色：成功    红色：未成功", ha="right", va="top", transform=ax.transAxes)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="评估生成的 rollouts.csv")
    parser.add_argument("--output", type=Path, help="PNG 路径；默认写到 CSV 所在目录")
    args = parser.parse_args()
    output = args.output or args.csv.with_name("rollout_results.png")
    rows = load_rollouts(args.csv)
    plot_rollouts(rows, output)
    print(f"已绘制 {len(rows)} 次评估、{sum(row['success'] for row in rows)} 次成功：{output.resolve()}")


if __name__ == "__main__":
    main()
