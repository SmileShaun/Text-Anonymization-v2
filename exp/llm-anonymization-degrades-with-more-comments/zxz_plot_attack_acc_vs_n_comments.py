#!/usr/bin/env python3
"""Plot Attack Acc vs. comment-count bins: original L0 vs final anonymized L3.

Reads both:
  results/synthpai_llm-anonymization_250pers/deepseek-chat-full-comments
  results/synthpai_remaining_50pers/deepseek-chat-full-comments

Outputs under results/_plots/:
  - attack_acc_vs_n_comments_L0_vs_L3.png / .pdf
  - attack_acc_vs_n_comments_L0_vs_L3.csv
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = REPO_ROOT / "results"
RUNS = (
    ("synthpai_llm-anonymization_250pers", "250pers"),
    ("synthpai_remaining_50pers", "50pers"),
)
BIN_EDGES = [0, 20, 40, 60, 80, 100, 120, 140, 10**9]
BIN_LABELS = ["1–20", "21–40", "41–60", "61–80", "81–100", "101–120", "121–140", "141+"]

C_ORIG = "#1D81A2"
C_ANON = "#EA985F"
C_DELTA = "#E74C41"
C_DELTA3 = "#A569BD"


def _setup_cjk_font() -> None:
    preferred = [
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Droid Sans Fallback",
        "Noto Sans CJK JP",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def count_comments(inference_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(inference_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            n = 0
            for g in obj.get("comments", []):
                if isinstance(g, dict) and "comments" in g:
                    n += len(g["comments"])
            counts[obj["username"]] = n
    return counts


def parse_is_correct(x) -> list:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return ast.literal_eval(x)


def load_eval_rows(results_root: Path) -> pd.DataFrame:
    frames = []
    for run_dir, split in RUNS:
        d = results_root / run_dir / "deepseek-chat-full-comments"
        eval_path = d / "eval_deepseek-chat_out.csv"
        inf_path = d / "inference_0.jsonl"
        if not eval_path.exists() or not inf_path.exists():
            raise FileNotFoundError(f"missing inputs under {d}")

        n_map = count_comments(inf_path)
        eval_df = pd.read_csv(eval_path)
        eval_df["ic"] = eval_df["is_correct"].apply(parse_is_correct)
        eval_df["top1"] = eval_df["ic"].apply(lambda x: int(x[0]) if x else 0)
        eval_df["top3"] = eval_df["ic"].apply(
            lambda x: int(any(int(v) == 1 for v in x))
        )
        eval_df = eval_df[eval_df["anon_level"].isin([0, 3])].copy()
        eval_df["n_comments"] = eval_df["id"].map(n_map)
        eval_df["split"] = split
        eval_df = eval_df.dropna(subset=["n_comments"])
        frames.append(eval_df)
    return pd.concat(frames, ignore_index=True)


def summarize_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = pd.cut(
        df["n_comments"], bins=BIN_EDGES, labels=BIN_LABELS, right=True
    )
    rows = []
    for b in BIN_LABELS:
        sub = df[df["bin"] == b]
        if sub.empty:
            continue
        n_users = int(sub["id"].nunique())
        for lvl, name in [(0, "original_L0"), (3, "anonymized_L3")]:
            s = sub[sub["anon_level"] == lvl]
            if s.empty:
                continue
            rows.append(
                {
                    "bin": b,
                    "level": name,
                    "anon_level": lvl,
                    "n_users": n_users,
                    "n_evals": len(s),
                    "top1": float(s["top1"].mean()),
                    "top3": float(s["top3"].mean()),
                    "top1_correct": int(s["top1"].sum()),
                    "top3_correct": int(s["top3"].sum()),
                }
            )
    return pd.DataFrame(rows)


def wide_table(sum_df: pd.DataFrame) -> pd.DataFrame:
    present = [b for b in BIN_LABELS if b in set(sum_df["bin"])]
    rows = []
    for b in present:
        o = sum_df[(sum_df["bin"] == b) & (sum_df["level"] == "original_L0")].iloc[0]
        a = sum_df[(sum_df["bin"] == b) & (sum_df["level"] == "anonymized_L3")].iloc[0]
        rows.append(
            {
                "bin": b,
                "n_users": int(o["n_users"]),
                "n_evals_L0": int(o["n_evals"]),
                "n_evals_L3": int(a["n_evals"]),
                "top1_original_L0": o["top1"],
                "top1_anonymized_L3": a["top1"],
                "delta_top1": o["top1"] - a["top1"],
                "top3_original_L0": o["top3"],
                "top3_anonymized_L3": a["top3"],
                "delta_top3": o["top3"] - a["top3"],
            }
        )
    return pd.DataFrame(rows)


def plot(wide: pd.DataFrame, out_png: Path, out_pdf: Path, n_users_total: int) -> None:
    _setup_cjk_font()
    present = list(wide["bin"])
    x = np.arange(len(present))
    orig_t1 = wide["top1_original_L0"].tolist()
    anon_t1 = wide["top1_anonymized_L3"].tolist()
    orig_t3 = wide["top3_original_L0"].tolist()
    anon_t3 = wide["top3_anonymized_L3"].tolist()
    delta_t1 = wide["delta_top1"].tolist()
    delta_t3 = wide["delta_top3"].tolist()
    n_users = wide["n_users"].tolist()
    n_evals = wide["n_evals_L0"].tolist()

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2.2, 1.1], "hspace": 0.32}
    )

    ax = axes[0]
    ax.plot(
        x, orig_t1, "o-", color=C_ORIG, lw=2.2, ms=8,
        label="原始 comments (L0) · top-1", zorder=3,
    )
    ax.plot(
        x, anon_t1, "s--", color=C_ANON, lw=2.2, ms=8,
        label="匿名后 comments (L3) · top-1", zorder=3,
    )
    ax.fill_between(x, orig_t1, anon_t1, color=C_DELTA, alpha=0.18, label="Δ (L0 − L3)", zorder=1)
    ax.plot(x, orig_t3, "o:", color=C_ORIG, lw=1.4, ms=5, alpha=0.55, label="原始 · top-3")
    ax.plot(x, anon_t3, "s:", color=C_ANON, lw=1.4, ms=5, alpha=0.55, label="匿名 · top-3")

    for i, (o, a, d) in enumerate(zip(orig_t1, anon_t1, delta_t1)):
        ax.annotate(
            f"Δ={d:+.1%}",
            xy=(i, (o + a) / 2),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=C_DELTA,
            va="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={nu})" for b, nu in zip(present, n_users)])
    ax.set_ylabel("Attack Accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, axis="y", alpha=0.35, linestyle="--")
    ax.set_title(
        "Attack Acc vs. Comment 数量（原始 L0 vs 最后一轮匿名 L3）\n"
        "合并 synthpai 250pers + remaining 50pers · deepseek-chat-full-comments · micro-avg",
        fontsize=12,
        pad=10,
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, ncol=2)

    ax2 = axes[1]
    w = 0.36
    b1 = ax2.bar(x - w / 2, delta_t1, width=w, color=C_DELTA, alpha=0.85, label="Δ top-1 (L0−L3)")
    b2 = ax2.bar(x + w / 2, delta_t3, width=w, color=C_DELTA3, alpha=0.75, label="Δ top-3 (L0−L3)")
    ax2.axhline(0, color="#333333", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(present)
    ax2.set_xlabel("Comment 数量分箱（括号内为用户数，见上图）")
    ax2.set_ylabel("Accuracy Drop")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax2.grid(True, axis="y", alpha=0.35, linestyle="--")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax2.set_title("匿名化带来的 Attack Acc 下降幅度（正值 = 匿名后更难被攻击）", fontsize=11)

    for rect, val in zip(b1, delta_t1):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + (0.005 if val >= 0 else -0.012),
            f"{val:+.1%}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=7.5,
            color=C_DELTA,
        )
    for rect, val in zip(b2, delta_t3):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + (0.005 if val >= 0 else -0.012),
            f"{val:+.1%}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=7.5,
            color="#7D3C98",
        )

    fig.text(
        0.5,
        0.01,
        f"Source: eval_deepseek-chat_out.csv · comment counts from inference_0.jsonl · "
        f"total users={n_users_total} · L0 evals={sum(n_evals)}",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <results-root>/_plots",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (args.results_root / "_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval_rows(args.results_root)
    sum_df = summarize_by_bin(df)
    wide = wide_table(sum_df)

    stem = "attack_acc_vs_n_comments_L0_vs_L3"
    wide.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.6f")
    plot(
        wide,
        out_dir / f"{stem}.png",
        out_dir / f"{stem}.pdf",
        n_users_total=int(df["id"].nunique()),
    )
    print(wide.to_string(index=False))
    print(f"saved: {out_dir / (stem + '.png')}")
    print(f"saved: {out_dir / (stem + '.pdf')}")
    print(f"saved: {out_dir / (stem + '.csv')}")


if __name__ == "__main__":
    main()
