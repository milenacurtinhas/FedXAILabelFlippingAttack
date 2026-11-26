#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_top15_shap_global_bars_ci_notitle.py

Plota Top-N features por caso com barras de erro do intervalo de confiança (IC),
a partir dos outputs do summarize_shap_global_means.py.

VERSÃO FINAL:
  - SEM título
  - SEM label no eixo x
  - COM IC (se disponível)
  - IC com limite inferior clampado em 0 (evita eixo negativo)
  - Estilo compatível com template SBC

Entradas esperadas:
  *_global_means/shap_global_means.csv
  *_global_means/shap_global_means_all_clients.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Estilo SBC-friendly
# ============================================================

def set_sbc_style(font_size: int = 8) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": font_size,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })


# ============================================================
# Leitura + IC
# ============================================================

def detect_ci_columns(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    cols = set(df.columns.astype(str))
    pat_low = re.compile(r"^ci(\d+)_low$")
    pat_high = re.compile(r"^ci(\d+)_high$")

    lows, highs = {}, {}
    for c in cols:
        m1 = pat_low.match(c)
        m2 = pat_high.match(c)
        if m1:
            lows[m1.group(1)] = c
        if m2:
            highs[m2.group(1)] = c

    common = sorted(set(lows) & set(highs), key=lambda x: int(x), reverse=True)
    if not common:
        return None
    lvl = common[0]
    return lvl, lows[lvl], highs[lvl]


def read_feature_means_and_ci(csv_path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns or "mean" not in df.columns:
        raise RuntimeError(f"CSV inválido: {csv_path}")

    df["feature"] = df["feature"].astype(str)
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df = df.dropna(subset=["mean"])

    ci_info = detect_ci_columns(df)
    if ci_info is None:
        return df[["feature", "mean"]], None

    lvl, low_col, high_col = ci_info
    df[low_col] = pd.to_numeric(df[low_col], errors="coerce")
    df[high_col] = pd.to_numeric(df[high_col], errors="coerce")
    df = df.dropna(subset=[low_col, high_col])

    return df[["feature", "mean", low_col, high_col]], lvl


# ============================================================
# Plot
# ============================================================

def shorten(names: List[str], max_len: int = 24) -> List[str]:
    return [s if len(s) <= max_len else s[: max_len - 1] + "…" for s in names]


def plot_top_barh_with_ci(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    top_n: int = 15,
    ci_level: Optional[str] = None,
    fig_w: float = 3.35,
    fig_h: float = 2.60,
    axis_labels: bool = True,
) -> None:
    d = df.sort_values("mean", ascending=False).head(top_n)
    d = d.sort_values("mean", ascending=True)

    feats = shorten(d["feature"].tolist())
    mean_vals = d["mean"].to_numpy(dtype=float)

    xerr = None
    if ci_level is not None:
        lo = d[f"ci{ci_level}_low"].to_numpy(dtype=float)
        hi = d[f"ci{ci_level}_high"].to_numpy(dtype=float)

        # clamp inferior do IC em zero
        lo = np.maximum(0.0, lo)

        # garantir sem erros negativos
        left = np.maximum(0.0, mean_vals - lo)
        right = np.maximum(0.0, hi - mean_vals)

        xerr = np.vstack([left, right])

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)

    ax.barh(
        feats,
        mean_vals,
        xerr=xerr,
        capsize=2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )

    # Sem título
    ax.set_title("")

    ax.set_xlabel("Média de |valor SHAP|" if axis_labels else "")

    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

    # garante que eixo horizontal nunca comece em negativo
    ax.set_xlim(left=0.0)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def maybe_plot(in_csv: Path, out_dir: Path, name: str, top_n: int, fig_w: float, fig_h: float,
               axis_labels: bool = True):
    if not in_csv.exists():
        print(f"[SKIP] {name}")
        return

    df, ci_lvl = read_feature_means_and_ci(in_csv)
    out_pdf = out_dir / f"top{top_n}_{name}.pdf"
    out_png = out_dir / f"top{top_n}_{name}.png"

    plot_top_barh_with_ci(
        df=df,
        out_pdf=out_pdf,
        out_png=out_png,
        top_n=top_n,
        ci_level=ci_lvl,
        fig_w=fig_w,
        fig_h=fig_h,
        axis_labels=axis_labels,
    )
    print(f"[OK] {out_pdf.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--means-root", default=".")
    ap.add_argument("--out-dir", default="fig_top_features_ci")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--font-size", type=int, default=8)
    ap.add_argument("--fig-w", type=float, default=3.35)
    ap.add_argument("--fig-h", type=float, default=2.60)
    ap.add_argument("--no-axis-labels", action="store_true", help="Remove o rótulo do eixo x")
    args = ap.parse_args()

    set_sbc_style(font_size=args.font_size)

    root = Path(args.means_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("MC_DT", root / "MC_DC_global_means" / "shap_global_means.csv"),
        ("MF_DT_Uni", root / "MA_DC_iid_global_means" / "shap_global_means.csv"),
        ("MF_DT_NonUni", root / "MA_DC_noniid_global_means" / "shap_global_means.csv"),
        ("MF_DL_Uni", root / "MA_DL_iid_global_means" / "shap_global_means_all_clients.csv"),
        ("MF_DL_NonUni", root / "MA_DL_noniid_global_means" / "shap_global_means_all_clients.csv"),
    ]

    for name, csv in cases:
        maybe_plot(csv, out_dir, name, args.top_n, args.fig_w, args.fig_h,
                   axis_labels=not bool(args.no_axis_labels))

    print("\n[DONE] Figuras geradas em:", out_dir.resolve())


if __name__ == "__main__":
    main()