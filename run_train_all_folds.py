#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_train_all_folds.py (PADRAO B - OPCAO 2, CORRIGIDA)

Layout dos clientes montados:

mounted_clients/
  fold_0/
    clients/
      iid/
        client0/ client1/ client2/ client3/
      noniid/
        client0/ client1/ client2/ client3/
  fold_1/
    clients/
      iid/
        ...

cv/
  fold_0/
    dataset_central/
      data_train.csv
      data_test.csv
  fold_1/
    dataset_central/
      ...

Resultados:
training_results/
  k_fold0/
    central/  (best_model.npz, metrics_best.txt)
    iid/      (best.model.npz, metrics_summary.txt)
    noniid/   (best.model.npz, metrics_summary.txt)
  k_fold1/
    ...

Regras especiais:
- fed/ehms_fed.py deve ser rodado com sudo.
- Antes de rodar cada fed, apagar SOMENTE:
    fed/server/best.model.npz
    fed/server/metrics_summary.txt
- O fed/ehms_fed.py (MODIFICADO) é responsável por copiar os resultados finais
  para o out-dir informado.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ----------------------------
# utilitarios
# ----------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            print(f"[CLEAN] Removido: {path}")
        else:
            print(f"[CLEAN] Não existe: {path}")
    except Exception as ex:
        eprint(f"[WARN] Não consegui apagar {path}: {ex}")


def run_cmd(
    cmd: list[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    env: Optional[dict] = None,
) -> None:
    """
    Executa comando e falha se returncode != 0.
    """
    print("\n" + "=" * 80)
    print("[RUN]", " ".join(cmd))
    if cwd:
        print("[CWD]", str(cwd))
    print("=" * 80)

    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        env=env,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Comando falhou (returncode={p.returncode}): {' '.join(cmd)}")


def validate_clients_dir(clients_dir: Path, n_clients: int) -> None:
    if not clients_dir.exists():
        raise FileNotFoundError(f"[ERRO] clients_dir não existe: {clients_dir}")

    for i in range(n_clients):
        cdir = clients_dir / f"client{i}"
        if not cdir.exists():
            raise FileNotFoundError(f"[ERRO] Cliente faltando: {cdir}")

        train_csv = cdir / "data_train.csv"
        test_csv = cdir / "data_test.csv"
        if not train_csv.exists() or not test_csv.exists():
            raise FileNotFoundError(
                f"[ERRO] client{i} precisa de data_train.csv e data_test.csv em {cdir}"
            )


def fold_name(i: int) -> str:
    return f"fold_{i}"


def results_fold_name(i: int) -> str:
    return f"k_fold{i}"


# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--k", type=int, required=True, help="Número de folds (ex: 5).")
    ap.add_argument("--cv-root", default="cv", help="Raiz dos folds do CV (ex: cv/).")
    ap.add_argument("--mount-root", required=True, help="Raiz dos clientes montados (mounted_clients/).")
    ap.add_argument("--results-root", default="training_results", help="Pasta raiz de resultados.")
    ap.add_argument("--n-clients", type=int, default=4, help="Número de clientes (ex: 4).")

    ap.add_argument("--central-script", default="ehms_central.py", help="Script do treino central.")
    ap.add_argument("--fed-script", default="fed/ehms_fed.py", help="Script do treino federado (MininetFed).")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument("--timeout-central", type=int, default=0, help="Timeout central em segundos (0 = sem timeout).")
    ap.add_argument("--timeout-fed", type=int, default=0, help="Timeout federado em segundos (0 = sem timeout).")

    ap.add_argument("--attack-rate", type=float, default=0.0, help="Label flipping no cliente malicioso: 0.0, 0.3, 0.6 ou 1.0.")
    ap.add_argument("--malicious-client", type=int, default=0)
    ap.add_argument("--attack-seed", type=int, default=42)

    ap.add_argument("--skip-central", action="store_true")
    ap.add_argument("--skip-iid", action="store_true")
    ap.add_argument("--skip-noniid", action="store_true")

    ap.add_argument(
        "--fed-server-dir",
        default="fed/server",
        help="Pasta do servidor MininetFed (onde ficam best.model.npz e metrics_summary.txt).",
    )

    args = ap.parse_args()

    k = int(args.k)
    cv_root = Path(args.cv_root)
    mount_root = Path(args.mount_root)
    results_root = Path(args.results_root)
    n_clients = int(args.n_clients)

    central_script = Path(args.central_script)
    fed_script = Path(args.fed_script)
    fed_server_dir = Path(args.fed_server_dir)

    timeout_c = None if int(args.timeout_central) <= 0 else int(args.timeout_central)
    timeout_f = None if int(args.timeout_fed) <= 0 else int(args.timeout_fed)

    # valida scripts
    if not central_script.exists():
        raise FileNotFoundError(f"[ERRO] central-script não encontrado: {central_script}")
    if not fed_script.exists():
        raise FileNotFoundError(f"[ERRO] fed-script não encontrado: {fed_script}")

    results_root.mkdir(parents=True, exist_ok=True)

    # Arquivos do MininetFed que devem ser limpos (somente esses)
    fed_best = fed_server_dir / "best.model.npz"
    fed_metrics = fed_server_dir / "metrics_summary.txt"

    for fi in range(k):
        fold_dir = cv_root / fold_name(fi)
        dataset_central_dir = fold_dir / "dataset_central"

        if not dataset_central_dir.exists():
            raise FileNotFoundError(f"[ERRO] dataset_central não encontrado em: {dataset_central_dir}")

        # Estrutura de output do fold
        fold_out = results_root / results_fold_name(fi)
        out_central = fold_out / "central"
        out_iid = fold_out / "iid"
        out_noniid = fold_out / "noniid"

        out_central.mkdir(parents=True, exist_ok=True)
        out_iid.mkdir(parents=True, exist_ok=True)
        out_noniid.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"[FOLD {fi}] dataset={dataset_central_dir}  out={fold_out}")
        print("#" * 100)

        # ----------------------------
        # 1) CENTRAL
        # ----------------------------
        if not args.skip_central:
            cmd_central = [
                sys.executable, str(central_script),
                "--dataset-dir", str(dataset_central_dir),
                "--out-dir", str(out_central),
                "--seed", str(args.seed),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
            ]
            run_cmd(cmd_central, timeout=timeout_c)

            # Central deve produzir exatamente esses:
            c_best = out_central / "best_model.npz"
            c_met = out_central / "metrics_best.txt"
            if not c_best.exists() or not c_met.exists():
                raise FileNotFoundError(
                    "[ERRO] Central terminou, mas não encontrei best_model.npz/metrics_best.txt em "
                    f"{out_central}"
                )
            print(f"[OK] CENTRAL fold {fi}: {c_best.name}, {c_met.name}")
        else:
            print(f"[SKIP] CENTRAL fold {fi}")

        # ----------------------------
        # helper para rodar FED (iid/noniid)
        # ----------------------------
        def run_fed(mode: str, out_dir: Path) -> None:
            # mounted_clients/fold_i/clients/<mode>
            clients_dir = mount_root / fold_name(fi) / "clients" / mode

            # valida que existem client0..clientN com CSVs
            validate_clients_dir(clients_dir, n_clients)

            # limpar SOMENTE os dois outputs do server antes de rodar (conflito de output)
            safe_unlink(fed_best)
            safe_unlink(fed_metrics)

            # Executa fed com sudo.
            # IMPORTANTE: fed/ehms_fed.py MODIFICADO exige --out-dir e suporta --variant-tag.
            cmd_fed = [
                "sudo",
                "python3", str(fed_script),
                "--clients-dir", str(clients_dir),
                "--out-dir", str(out_dir),
                "--n-clients", str(n_clients),
                "--variant-tag", str(mode),
                "--attack-rate", str(args.attack_rate),
                "--malicious-client", str(args.malicious_client),
                "--attack-seed", str(args.attack_seed),
            ]

            run_cmd(cmd_fed, timeout=timeout_f)

            # O fed_script deve ter copiado os resultados para out_dir.
            out_best = out_dir / "best.model.npz"
            out_met = out_dir / "metrics_summary.txt"
            if not out_best.exists():
                raise FileNotFoundError(f"[ERRO] Não encontrei {out_best} após rodar federado ({mode})")
            if not out_met.exists():
                raise FileNotFoundError(f"[ERRO] Não encontrei {out_met} após rodar federado ({mode})")

            print(f"[OK] FED {mode} fold {fi}: resultados em {out_dir}")

        # ----------------------------
        # 2) FED IID
        # ----------------------------
        if not args.skip_iid:
            run_fed("iid", out_iid)
        else:
            print(f"[SKIP] FED IID fold {fi}")

        # ----------------------------
        # 3) FED NONIID
        # ----------------------------
        if not args.skip_noniid:
            run_fed("noniid", out_noniid)
        else:
            print(f"[SKIP] FED NONIID fold {fi}")

    print("\n" + "=" * 100)
    print("[DONE] Todos os folds concluídos.")
    print(f"[RESULTS] {results_root.resolve()}")
    print("=" * 100)


if __name__ == "__main__":
    main()