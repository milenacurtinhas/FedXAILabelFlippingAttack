#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_ecdf_cosine_distance.py

Gera gráficos ECDF da distância do cosseno (cosine_distance) entre cenários de
ataque por label flipping, no contexto MF_DL:

  MF_DL (Uni):    0% vs 30%, 0% vs 60%, 0% vs 100%   -> 3 gráficos
  MF_DL (NonUni): 0% vs 30%, 0% vs 60%, 0% vs 100%   -> 3 gráficos

Para cada gráfico, plota uma curva por fold.

REQUISITOS (conforme pedido):
- Padrão compatível com template SBC (fonte serif, tamanho pequeno, figura compacta)
- SEM título
- SEM labels de eixo
- Cores ALTAMENTE contrastantes e consistentes por fold entre todos os cenários
  (Fold 1 sempre a mesma cor em todos os cenários)
- Curvas levemente mais grossas
- Escala geométrica igual em X e Y (aspect='equal')
- fold i -> legenda "Fold i+1"
- Saída: PDF + PNG (300 dpi)

Entrada esperada: pasta raiz gerada por run_compare_attack_scenarios.py:
  compare_root/
    MF_DL_Uni_lf00_vs_lf30/fold0/cosine_distance.csv
    MF_DL_Uni_lf00_vs_lf30/fold1/cosine_distance.csv
    ...
    MF_DL_NonUni_lf00_vs_lf100/fold4/cosine_distance.csv

Saída (out_dir):
  ecdf_cosine_MF_DL_Uni_lf00_vs_lf30.pdf (+ png)
  ecdf_cosine_MF_DL_Uni_lf00_vs_lf60.pdf (+ png)
  ecdf_cosine_MF_DL_Uni_lf00_vs_lf100.pdf (+ png)
  ecdf_cosine_MF_DL_NonUni_lf00_vs_lf30.pdf (+ png)
  ecdf_cosine_MF_DL_NonUni_lf00_vs_lf60.pdf (+ png)
  ecdf_cosine_MF_DL_NonUni_lf00_vs_lf100.pdf (+ png)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
# ECDF
# ============================================================

def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    x = np.sort(v)
    y = np.arange(1, x.size + 1, dtype=np.float64) / float(x.size)
    return x, y


# ============================================================
# Leitura
# ============================================================

def load_cosine_values(csv_path: Path, col: str = "cosine_distance") -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"Não encontrei: {csv_path}")
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise RuntimeError(f"CSV sem coluna '{col}': {csv_path}")
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return vals


def fold_label(i: int) -> str:
    # fold i -> fold i+1
    return f"Fold {i + 1}"


# ============================================================
# Cores consistentes e contrastantes por fold
# ============================================================

def fold_colors(k: int) -> List:
    """
    Paleta FIXA e altamente contrastante (colorblind-safe).
    Base: Okabe–Ito (8 cores) — excelente para artigos.
    Se k > 8: completa com algumas cores extras do tab10.

    Observação: Fold i -> cor i (estável).
    """
    if k <= 0:
        return []

    # Okabe–Ito (8 cores, alto contraste e colorblind-safe)
    okabe_ito_hex = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ]

    def hex_to_rgb(h: str):
        h = h.lstrip("#")
        return (int(h[0:2], 16)/255.0, int(h[2:4], 16)/255.0, int(h[4:6], 16)/255.0)

    base = [hex_to_rgb(h) for h in okabe_ito_hex]

    if k <= len(base):
        return base[:k]

    # completa com tab10 (evitando repetir as já usadas)
    tab10 = list(plt.get_cmap("tab10").colors)
    extra = []
    for c in tab10:
        if c not in base:
            extra.append(c)

    colors = base + extra
    if k <= len(colors):
        return colors[:k]

    # fallback: repete ciclo (se alguém usar k muito grande)
    return [colors[i % len(colors)] for i in range(k)]



# ============================================================
# Plot por cenário
# ============================================================

def plot_ecdf_scenario(
    fold_to_vals: Dict[int, np.ndarray],
    out_pdf: Path,
    out_png: Path,
    k: int,
    colors: List,
    fig_w_in: float,
    fig_h_in: float,
    show_legend: bool = True,
    line_w: float = 1.6,
    axis_labels: bool = True,
) -> None:
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_subplot(111)

    # SEM título
    ax.set_title("")

    # curvas por fold (cores consistentes + mais grossas)
    for i in range(k):
        vals = fold_to_vals.get(i, None)
        if vals is None or vals.size == 0:
            continue
        x, y = ecdf(vals)
        if x.size == 0:
            continue
        ax.plot(
            x, y,
            linewidth=line_w,
            color=colors[i],
            label=fold_label(i),
        )

    # SEM labels de eixo
    ax.set_xlabel("Distância do cosseno" if axis_labels else "")
    ax.set_ylabel("Proporção acumulada" if axis_labels else "")

    # limites típicos da distância cosseno (1 - cos): [0, 2]
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)

    # ✅ mesma escala geométrica em x e y
    ax.set_aspect("equal", adjustable="box")

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    if show_legend:
        ax.legend(
            loc="lower right",
            frameon=False,
            ncol=2 if k >= 6 else 1,
            handlelength=1.6,
            columnspacing=0.8,
            labelspacing=0.2,
        )

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
    ap.add_argument("--compare-root", required=True, help="Pasta raiz das comparações (out_root do driver)")
    ap.add_argument("--k", type=int, required=True, help="Número de folds (k)")
    ap.add_argument("--out-dir", default="fig_ecdf_cosine", help="Pasta de saída das figuras")
    ap.add_argument("--font-size", type=int, default=8)

    # 1 coluna (SBC): ~3.35in de largura
    ap.add_argument("--fig-w", type=float, default=3.35)
    ap.add_argument("--fig-h", type=float, default=2.45)

    ap.add_argument("--no-legend", action="store_true", help="Remove legenda")
    ap.add_argument("--line-w", type=float, default=1.6, help="Espessura das curvas (default: 1.6)")
    ap.add_argument("--no-axis-labels", action="store_true", help="Remove os rótulos dos eixos")
    ap.add_argument("--contexts", default="Uni,NonUni", help="Contextos MF_DL a plotar")
    ap.add_argument("--baseline", default="00", help="Sufixo do cenário sem ataque")
    ap.add_argument("--attacks", default="30,60,100", help="Sufixos dos cenários com ataque")
    args = ap.parse_args()

    compare_root = Path(args.compare_root)
    if not compare_root.exists():
        raise FileNotFoundError(f"compare-root não existe: {compare_root}")

    k = int(args.k)
    if k <= 0:
        raise RuntimeError("--k precisa ser > 0")

    set_sbc_style(font_size=int(args.font_size))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cores consistentes por fold para todos os cenários (alto contraste)
    colors = fold_colors(k)

    # 1 gráfico por (contexto, ataque): nome da pasta -> tag de saída (iguais)
    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    scenarios: List[Tuple[str, str]] = []
    for ctx in contexts:
        for atk in attacks:
            name = f"MF_DL_{ctx}_lf{args.baseline}_vs_lf{atk}"
            scenarios.append((name, name))

    for folder_name, out_tag in scenarios:
        scen_dir = compare_root / folder_name
        if not scen_dir.exists():
            print(f"[SKIP] Cenário não encontrado: {scen_dir}")
            continue

        fold_to_vals: Dict[int, np.ndarray] = {}

        for i in range(k):
            csv_path = scen_dir / f"fold{i}" / "cosine_distance.csv"
            if not csv_path.exists():
                print(f"[WARN] faltando: {csv_path}")
                continue
            fold_to_vals[i] = load_cosine_values(csv_path, col="cosine_distance")

        out_pdf = out_dir / f"ecdf_cosine_{out_tag}.pdf"
        out_png = out_dir / f"ecdf_cosine_{out_tag}.png"

        plot_ecdf_scenario(
            fold_to_vals=fold_to_vals,
            out_pdf=out_pdf,
            out_png=out_png,
            k=k,
            colors=colors,
            fig_w_in=float(args.fig_w),
            fig_h_in=float(args.fig_h),
            show_legend=not bool(args.no_legend),
            line_w=float(args.line_w),
            axis_labels=not bool(args.no_axis_labels),
        )

        print(f"[OK] {out_pdf.name} (+ PNG)")

    print("\n[DONE] ECDFs gerados em:", out_dir.resolve())


if __name__ == "__main__":
    main()