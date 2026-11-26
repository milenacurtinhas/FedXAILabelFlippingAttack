#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_pmf_topk_sign_consistency_informative_bins_fixedcolors.py

PMF do "número de sinais que batem" no TOP-k:
  consistency ∈ [0,1]  =>  s = round(k * consistency) ∈ [0,k]

Plota SOMENTE bins "informativos":
- Um bin agregado "< s_min"
- Bins individuais s_min..k
onde s_min = ceil(threshold*k) (default 0.6)

Agrega sobre folds: média + IC 0.95 (t-interval; amostras = folds)

IMPORTANTE: usa o MESMO mapeamento de cores por cenário do script de Jaccard.

Entrada (gerada por run_compare_attack_scenarios.py):
  compare_root/MF_DL_<Uni|NonUni>_lf00_vs_lf<30|60|100>/fold<i>/topk_sign_consistency_<k>.csv

Comparações (uma barra por comparação, em cada bin):
  0% vs 30%, 0% vs 60%, 0% vs 100% de ataque por label flipping,
  em figuras separadas para MF_DL (Uni) e MF_DL (NonUni).

Saídas (uma figura por contexto e por k):
  out_dir/pmf_sign_informative_MF_DL_Uni_top5.pdf (+ png)
  out_dir/pmf_sign_informative_MF_DL_Uni_top10.pdf (+ png)
  out_dir/pmf_sign_informative_MF_DL_Uni_top15.pdf (+ png)
  out_dir/pmf_sign_informative_MF_DL_NonUni_top5.pdf (+ png)  ... idem para top10/top15

NaN:
- por padrão, ignora NaN (overlap vazio)
- opcional: --nan-as-zero para tratar NaN como 0.0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy opcional
try:
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


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
# CORES FIXAS (cenários de ataque) — IGUAL NOS DOIS SCRIPTS DE PMF
# ============================================================
# Cada barra compara o cenário SEM ataque (0%) com um cenário COM ataque.
# A cor segue a intensidade do ataque (rampa azul clara -> escura, uma só matiz),
# o que também funciona em impressão em tons de cinza.

ATTACK_LABELS = {
    "30": "0% vs 30%",
    "60": "0% vs 60%",
    "100": "0% vs 100%",
}

ATTACK_COLORS = {
    "30": "#6da7ec",
    "60": "#256abf",
    "100": "#0d366b",
}

ATTACKS_ORDER = ["30", "60", "100"]

def attack_color(key: str):
    if key not in ATTACK_COLORS:
        raise KeyError(f"Cenário de ataque sem cor definida: {key}")
    return ATTACK_COLORS[key]


# ============================================================
# Estatística: mean + IC
# ============================================================

def t_crit(ci_level: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    alpha = 1.0 - float(ci_level)
    if _HAS_SCIPY:
        return float(student_t.ppf(1.0 - alpha / 2.0, df=df))
    if abs(ci_level - 0.95) < 1e-9:
        return 1.96
    return 1.96


def mean_ci(values: np.ndarray, ci_level: float = 0.95) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(v))
    if n == 1:
        return m, m, m
    s = float(np.std(v, ddof=1))
    sem = s / math.sqrt(n)
    tc = t_crit(ci_level=ci_level, df=n - 1)
    half = tc * sem if np.isfinite(tc) else float("nan")
    return m, (m - half), (m + half)


# ============================================================
# Consistency -> número de sinais iguais (inteiro s)
# ============================================================

def consistency_to_sign_matches(cons: np.ndarray, k: int, nan_as_zero: bool) -> np.ndarray:
    c = np.asarray(cons, dtype=np.float64)
    if nan_as_zero:
        c = np.nan_to_num(c, nan=0.0)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return np.array([], dtype=int)
    c = np.clip(c, 0.0, 1.0)
    s = np.rint(k * c).astype(int)
    return np.clip(s, 0, k)


# ============================================================
# PMF com bins informativos
# ============================================================

def pmf_informative_bins(s_int: np.ndarray, k: int, s_min: int) -> Tuple[np.ndarray, List[str]]:
    labels = [f"<{s_min}"] + [str(x) for x in range(s_min, k + 1)]

    s_int = np.asarray(s_int, dtype=int)
    s_int = s_int[(s_int >= 0) & (s_int <= k)]
    if s_int.size == 0:
        probs = np.full((len(labels),), np.nan, dtype=np.float64)
        return probs, labels

    total = float(s_int.size)
    probs = [float(np.sum(s_int < s_min) / total)]
    for s in range(s_min, k + 1):
        probs.append(float(np.sum(s_int == s) / total))

    return np.array(probs, dtype=np.float64), labels


# ============================================================
# IO
# ============================================================

def read_consistency_values(csv_path: Path, col: str) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"Não encontrei: {csv_path}")
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise RuntimeError(f"CSV sem coluna '{col}': {csv_path}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)  # pode conter NaN


# ============================================================
# Plot
# ============================================================

def plot_pmf_grouped(
    x_labels: List[str],
    scenario_to_stats: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_pdf: Path,
    out_png: Path,
    fig_w_in: float,
    fig_h_in: float,
    show_legend: bool,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    bins = np.arange(len(x_labels), dtype=np.float64)

    keys_in_plot = [k for k in ATTACKS_ORDER if k in scenario_to_stats]
    n_scen = len(keys_in_plot)

    group_width = 0.86
    bar_w = group_width / max(1, n_scen)
    offsets = (np.arange(n_scen) - (n_scen - 1) / 2.0) * bar_w

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_subplot(111)
    ax.set_title("")

    for si, skey in enumerate(keys_in_plot):
        mean_p, lo, hi = scenario_to_stats[skey]
        yerr = np.vstack([
            np.maximum(0.0, mean_p - lo),
            np.maximum(0.0, hi - mean_p),
        ])

        x = bins + offsets[si]
        ax.bar(
            x,
            mean_p,
            width=bar_w * 0.95,
            yerr=yerr,
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            color=attack_color(skey),
            linewidth=0.0,
            label=ATTACK_LABELS[skey],
        )

    ax.set_xticks(bins)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    if show_legend:
        ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.6, labelspacing=0.2)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-root", required=True)
    ap.add_argument("--kfolds", type=int, required=True)
    ap.add_argument("--out-dir", default="fig_pmf_sign_informative")
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--font-size", type=int, default=8)

    # SBC 1 coluna
    ap.add_argument("--fig-w", type=float, default=3.35)
    ap.add_argument("--fig-h", type=float, default=2.45)

    ap.add_argument("--no-legend", action="store_true")
    ap.add_argument("--no-axis-labels", action="store_true", help="Remove os rótulos dos eixos")
    ap.add_argument("--contexts", default="Uni,NonUni", help="Contextos MF_DL a plotar")
    ap.add_argument("--baseline", default="00", help="Sufixo do cenário sem ataque")
    ap.add_argument("--attacks", default="30,60,100", help="Sufixos dos cenários com ataque")
    ap.add_argument("--nan-as-zero", action="store_true",
                    help="Trata NaN (overlap vazio) como 0.0 antes de converter para s.")
    args = ap.parse_args()

    compare_root = Path(args.compare_root)
    if not compare_root.exists():
        raise FileNotFoundError(f"compare-root não existe: {compare_root}")

    kfolds = int(args.kfolds)
    if kfolds <= 0:
        raise RuntimeError("--kfolds precisa ser > 0")

    thr = float(args.threshold)
    if not (0.0 < thr <= 1.0):
        raise RuntimeError("--threshold deve estar em (0,1]. Ex: 0.6")

    set_sbc_style(font_size=int(args.font_size))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    for a in attacks:
        if a not in ATTACK_COLORS:
            raise RuntimeError(f"--attacks: '{a}' sem cor definida (use {', '.join(ATTACKS_ORDER)})")

    for ctx, topk in [(c, t) for c in contexts for t in (5, 10, 15)]:
        col = f"topk_sign_consistency_{topk}"
        s_min = int(math.ceil(thr * topk))

        x_labels = [f"<{s_min}"] + [str(x) for x in range(s_min, topk + 1)]
        nbins = len(x_labels)

        scenario_to_stats: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for skey in attacks:
            scen_dir = compare_root / f"MF_DL_{ctx}_lf{args.baseline}_vs_lf{skey}"
            if not scen_dir.exists():
                print(f"[SKIP] Cenário não encontrado: {scen_dir}")
                continue

            pmfs: List[np.ndarray] = []
            for fi in range(kfolds):
                csv_path = scen_dir / f"fold{fi}" / f"topk_sign_consistency_{topk}.csv"
                if not csv_path.exists():
                    print(f"[WARN] faltando: {csv_path}")
                    continue

                cons = read_consistency_values(csv_path, col=col)
                s_int = consistency_to_sign_matches(cons, k=topk, nan_as_zero=bool(args.nan_as_zero))
                pmf, _ = pmf_informative_bins(s_int, k=topk, s_min=s_min)

                if np.any(~np.isfinite(pmf)) or pmf.size != nbins:
                    print(f"[WARN] PMF inválida em {csv_path}. Pulando fold.")
                    continue

                pmfs.append(pmf)

            if not pmfs:
                print(f"[SKIP] Sem folds válidos para {ctx} 0% vs {skey}% em topk={topk}")
                continue

            M = np.stack(pmfs, axis=0)  # (n_folds, nbins)

            mean_p = np.zeros((nbins,), dtype=np.float64)
            lo = np.zeros((nbins,), dtype=np.float64)
            hi = np.zeros((nbins,), dtype=np.float64)

            for b in range(nbins):
                m, l, h = mean_ci(M[:, b], ci_level=float(args.ci_level))
                mean_p[b], lo[b], hi[b] = m, l, h

            scenario_to_stats[skey] = (mean_p, lo, hi)

        if not scenario_to_stats:
            print(f"[SKIP] Nenhum cenário válido para {ctx} topk={topk}")
            continue

        out_pdf = out_dir / f"pmf_sign_informative_MF_DL_{ctx}_top{topk}.pdf"
        out_png = out_dir / f"pmf_sign_informative_MF_DL_{ctx}_top{topk}.png"

        plot_pmf_grouped(
            x_labels=x_labels,
            scenario_to_stats=scenario_to_stats,
            out_pdf=out_pdf,
            out_png=out_png,
            fig_w_in=float(args.fig_w),
            fig_h_in=float(args.fig_h),
            show_legend=not bool(args.no_legend),
            xlabel="" if args.no_axis_labels else f"Sinais consistentes no top-{topk}",
            ylabel="" if args.no_axis_labels else "Probabilidade",
        )

        print(f"[OK] {out_pdf.name} (+ PNG) | topk={topk} s_min={s_min}")

    print("\n[DONE] Sign-consistency PMF informativa gerada em:", out_dir.resolve())


if __name__ == "__main__":
    main()