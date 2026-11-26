#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_compare_attack_scenarios.py

Compara as explicações SHAP do contexto MF_DL (modelo federado aplicado aos dados
locais de cada cliente) ENTRE cenários de ataque por label flipping:

    0% de ataque  vs  30% de ataque
    0% de ataque  vs  60% de ataque
    0% de ataque  vs 100% de ataque

separadamente para MF_DL (Uni) e MF_DL (NonUni), em cada fold.

Usa o mesmo compare_shap_instancewise.py do fluxo original (métricas por instância,
alinhadas por instance_id). Como os dados de teste dos clientes são os mesmos em
todos os cenários (o ataque altera só os rótulos de treino), as instâncias batem.

Entrada (gerada pelo run_shap_all_folds.py, uma pasta por cenário):
  <shap-prefix><baseline>/k_fold<i>/MA_DL_iid/data_test_shap_values_all_clients.csv
  <shap-prefix><baseline>/k_fold<i>/MA_DL_noniid/data_test_shap_values_all_clients.csv
  <shap-prefix><ataque>/k_fold<i>/MA_DL_.../data_test_shap_values_all_clients.csv

Saída:
  <out-root>/MF_DL_Uni_lf00_vs_lf30/fold<i>/{cosine_distance.csv, jaccard_topk_*.csv, ...}
  <out-root>/MF_DL_Uni_lf00_vs_lf60/fold<i>/...
  <out-root>/MF_DL_Uni_lf00_vs_lf100/fold<i>/...
  <out-root>/MF_DL_NonUni_lf00_vs_lf30/fold<i>/...
  ...

Exemplo:
  python3 run_compare_attack_scenarios.py --k 5 --shap-prefix shap_lf \\
    --baseline 00 --attacks 30,60,100 --out-root comparisons_attack
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# pasta MA_DL_* do run_shap_all_folds.py -> nome do contexto usado no artigo
VARIANTS = {
    "iid": "Uni",
    "noniid": "NonUni",
}

MERGED_FILE = "data_test_shap_values_all_clients.csv"


def run_cmd(cmd: list[str], timeout: Optional[int] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout)


def comparison_name(context: str, baseline: str, attack: str) -> str:
    """Ex.: ('Uni', '00', '30') -> 'MF_DL_Uni_lf00_vs_lf30'"""
    return f"MF_DL_{context}_lf{baseline}_vs_lf{attack}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="Número de folds")
    ap.add_argument("--shap-prefix", default="shap_lf",
                    help="Prefixo das pastas SHAP de cada cenário (ex.: shap_lf -> shap_lf00, shap_lf30, ...)")
    ap.add_argument("--baseline", default="00", help="Sufixo do cenário sem ataque (default: 00)")
    ap.add_argument("--attacks", default="30,60,100", help="Sufixos dos cenários com ataque")
    ap.add_argument("--variants", default="iid,noniid", help="Variantes MA_DL a comparar (iid,noniid)")
    ap.add_argument("--out-root", default="comparisons_attack", help="Pasta raiz de saída")
    ap.add_argument("--compare-script", default="compare_shap_instancewise.py")
    ap.add_argument("--rbo-p", type=float, default=0.9)
    ap.add_argument("--topk", default="5,10,15")
    ap.add_argument("--timeout", type=int, default=0)
    args = ap.parse_args()

    k = int(args.k)
    baseline = str(args.baseline)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    compare_script = Path(args.compare_script)
    out_root = Path(args.out_root)
    timeout = None if int(args.timeout) <= 0 else int(args.timeout)

    if not compare_script.exists():
        raise FileNotFoundError(f"compare-script não encontrado: {compare_script}")
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"Variante desconhecida: {v} (use: {', '.join(VARIANTS)})")

    base_root = Path(f"{args.shap_prefix}{baseline}")
    for root in [base_root] + [Path(f"{args.shap_prefix}{a}") for a in attacks]:
        if not root.exists():
            raise FileNotFoundError(f"Pasta SHAP do cenário não existe: {root}")

    out_root.mkdir(parents=True, exist_ok=True)

    for i in range(k):
        for variant in variants:
            context = VARIANTS[variant]
            a_path = base_root / f"k_fold{i}" / f"MA_DL_{variant}" / MERGED_FILE
            if not a_path.exists():
                raise FileNotFoundError(f"[baseline] arquivo não encontrado: {a_path}")

            for attack in attacks:
                b_path = Path(f"{args.shap_prefix}{attack}") / f"k_fold{i}" / f"MA_DL_{variant}" / MERGED_FILE
                if not b_path.exists():
                    raise FileNotFoundError(f"[ataque {attack}] arquivo não encontrado: {b_path}")

                out_dir = out_root / comparison_name(context, baseline, attack) / f"fold{i}"
                out_dir.mkdir(parents=True, exist_ok=True)

                run_cmd([
                    sys.executable, str(compare_script),
                    "--a", str(a_path),
                    "--b", str(b_path),
                    "--out-dir", str(out_dir),
                    "--rbo-p", str(args.rbo_p),
                    "--topk", str(args.topk),
                ], timeout=timeout)

    print("\n[DONE] Comparações entre cenários de ataque concluídas.")
    print(f"[OUT] {out_root.resolve()}")


if __name__ == "__main__":
    main()
