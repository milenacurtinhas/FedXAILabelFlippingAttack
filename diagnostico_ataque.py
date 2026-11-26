#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostico_ataque.py

Junta, por fold, distribuição (Uni/NonUni) e taxa de ataque, as métricas de desempenho
do modelo federado e as estatísticas das explicações SHAP (contexto MF_DL), para testar
se a divergência das explicações acompanha o dano ao desempenho.

Para cada (distribuição, fold, taxa p) calcula:
  - desempenho (de results_lf<p>/k_fold<i>/<iid|noniid>/metrics_summary.txt):
      acc, f1 (macro), precisão e recall da classe ataque, % de predições "ataque"
  - explicações (de shap_lf<p>/k_fold<i>/MA_DL_<iid|noniid>/data_test_shap_values_all_clients.csv):
      phi0_medio     = média de base_value (valor base do SHAP, em logit)
      fx_medio       = média de f_x (logit de saída)
      norma_phi      = média de ||phi(x)|| (magnitude das explicações)
  - divergência em relação a 0% (de comparisons_attack/MF_DL_<ctx>_lf00_vs_lf<p>/fold<i>/cosine_distance.csv):
      cos_mediana, cos_p90

Saídas:
  diagnostico_ataque.csv   (uma linha por distribuição x fold x taxa)
  resumo impresso na tela, incluindo correlações de Spearman entre a divergência
  (cos_mediana) e as variações de desempenho e de phi0 em relação a 0%.

Uso (na raiz do projeto):
  python3 diagnostico_ataque.py --k 5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = {"iid": "Uni", "noniid": "NonUni"}
MERGED = "data_test_shap_values_all_clients.csv"


def read_metrics(path: Path) -> dict:
    """Lê o metrics_summary.txt do MininetFed (relatório GLOBAL)."""
    if not path.exists():
        return {}
    t = path.read_text(encoding="utf-8", errors="replace")

    def scalar(key: str) -> float:
        m = re.search(rf"-\s*{key}\s*:\s*([0-9.]+)", t)
        return float(m.group(1)) if m else float("nan")

    out = {"acc": scalar("accuracy"), "f1_macro": scalar("f1")}
    m = re.search(r"true_0\s+(\d+)\s+(\d+)\s+true_1\s+(\d+)\s+(\d+)", t)
    if m:
        tn, fp, fn, tp = map(int, m.groups())
        n = tn + fp + fn + tp
        out["prec_ataque"] = tp / (tp + fp) if (tp + fp) else float("nan")
        out["rec_ataque"] = tp / (tp + fn) if (tp + fn) else float("nan")
        out["pct_pred_ataque"] = 100.0 * (tp + fp) / n
        out["fpr"] = 100.0 * fp / (tn + fp) if (tn + fp) else float("nan")
    return out


def read_shap_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    out = {"norma_phi": float(np.linalg.norm(df[shap_cols].to_numpy(dtype=float), axis=1).mean())}
    if "base_value" in df.columns:
        out["phi0_medio"] = float(df["base_value"].mean())
    if "f_x" in df.columns:
        out["fx_medio"] = float(df["f_x"].mean())
    return out


def read_cosine(path: Path) -> dict:
    if not path.exists():
        return {}
    v = pd.read_csv(path)["cosine_distance"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return {"cos_mediana": float(np.median(v)), "cos_p90": float(np.percentile(v, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rates", default="00,30,60,100", help="Sufixos dos cenários (o primeiro é a referência)")
    ap.add_argument("--results-prefix", default="results_lf")
    ap.add_argument("--shap-prefix", default="shap_lf")
    ap.add_argument("--compare-root", default="comparisons_attack")
    ap.add_argument("--out", default="diagnostico_ataque.csv")
    args = ap.parse_args()

    rates = [r.strip() for r in args.rates.split(",") if r.strip()]
    base = rates[0]
    rows = []

    for variant, ctx in VARIANTS.items():
        for i in range(args.k):
            for r in rates:
                row = {"dist": ctx, "fold": i + 1, "p": int(r)}
                row.update(read_metrics(Path(f"{args.results_prefix}{r}") / f"k_fold{i}" / variant / "metrics_summary.txt"))
                row.update(read_shap_stats(Path(f"{args.shap_prefix}{r}") / f"k_fold{i}" / f"MA_DL_{variant}" / MERGED))
                if r != base:
                    row.update(read_cosine(Path(args.compare_root) / f"MF_DL_{ctx}_lf{base}_vs_lf{r}" / f"fold{i}" / "cosine_distance.csv"))
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Nada encontrado: confira se está rodando na raiz do projeto.")

    # variações em relação ao cenário de referência (mesmo fold e distribuição)
    ref = df[df.p == int(base)].set_index(["dist", "fold"])
    for col in ["acc", "f1_macro", "pct_pred_ataque", "phi0_medio", "norma_phi"]:
        if col in df.columns:
            df[f"delta_{col}"] = df.apply(
                lambda x: x[col] - ref.loc[(x["dist"], x["fold"]), col]
                if (x["dist"], x["fold"]) in ref.index else np.nan, axis=1)

    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    show = [c for c in ["dist", "fold", "p", "f1_macro", "pct_pred_ataque", "phi0_medio",
                        "norma_phi", "cos_mediana", "cos_p90"] if c in df.columns]
    print(df[show].round(3).to_string(index=False))

    # correlação de Spearman entre divergência e variações, por distribuição (taxas > 0)
    att = df[df.p != int(base)]
    if "cos_mediana" in att.columns:
        print("\nCorrelação de Spearman com cos_mediana (taxas com ataque, 15 pontos por distribuição):")
        for ctx in VARIANTS.values():
            x = att[att.dist == ctx]
            for col in ["delta_f1_macro", "delta_pct_pred_ataque", "delta_phi0_medio", "delta_norma_phi"]:
                if col in x.columns and x[col].notna().sum() >= 4:
                    rho = x["cos_mediana"].corr(x[col].abs(), method="spearman")
                    print(f"  {ctx:6s} |{col}|: rho = {rho:+.2f}")

    print(f"\n[OK] {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()