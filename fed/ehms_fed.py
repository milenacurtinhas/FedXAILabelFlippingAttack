#!/usr/bin/env python3
"""
fed/ehms_fed.py (MODIFICADO)

Agora aceita:
- --clients-dir: pasta com client0..client3 (montados para o fold/variant)
- --out-dir: pasta para copiar os resultados finais (best.model.npz e metrics_summary.txt)
- --n-clients: default 4
- --variant-tag: só para log (iid/noniid)

Exemplo (RODAR COM SUDO):
sudo -E python3 fed/ehms_fed.py \
  --clients-dir mounted_clients/fold_0/clients/iid \
  --out-dir training_results/k_fold0/iid \
  --n-clients 4 \
  --variant-tag iid
"""

import argparse
import shutil
from pathlib import Path

from mininetfed.core.dto.metrics import MetricType
from mininetfed.core.fed_options import ServerOptions, ClientAcceptorType, ClientSelectorType, AggregatorType
from mininetfed.sim.net import MininetFed
from mininetfed.sim.nodes import FedServerNode, FedClientNode, FedBrokerNode
from mininetfed.sim.util.docker_utils import build_fed_node_docker_image

def cleanup_fed_server_outputs():
    files_to_remove = [
        Path("fed/server/best.model.npz"),
        Path("fed/server/metrics_summary.txt"),
    ]

    for f in files_to_remove:
        if f.exists():
            f.unlink()
            print(f"[CLEAN] Removido: {f}")
        else:
            print(f"[CLEAN] Não existe: {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients-dir", required=True, help="Pasta com client0..clientN (montados)")
    ap.add_argument("--out-dir", required=True, help="Pasta para copiar best.model.npz e metrics_summary.txt")
    ap.add_argument("--n-clients", type=int, default=4)
    ap.add_argument("--variant-tag", default="fed", help="Somente tag para log (iid/noniid)")
    ap.add_argument("--attack-rate", type=float, default=0.0, help="Fração dos dados de treino do cliente malicioso com rótulo invertido (0.0, 0.3, 0.6, 1.0). 0.0 = sem ataque")
    ap.add_argument("--malicious-client", type=int, default=0, help="Índice do ciente malicioso (o mesmo em todos os cenários)")
    ap.add_argument("--attack-seed", type=int, default=42, help="Semente para escolher quais amostras têm o rótulo invertido")
    args = ap.parse_args()

    if not 0.0 <= args.attack_rate <= 1.0:
        raise ValueError(f"--attack-rate deve estar em [0, 1]; recebido {args.attack_rate}")
    if not 0 <= args.malicious_client < args.n_clients:
        raise ValueError(f"--malicious-client deve estar em [0, {args.n_clients - 1}]")

    clients_dir = Path(args.clients_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clients_dir.exists():
        raise FileNotFoundError(f"clients-dir não existe: {clients_dir}")

    # server_args (mesmo que você tinha)
    server_args = {
        ServerOptions.MIN_CLIENTS      : args.n_clients,
        ServerOptions.NUM_ROUNDS       : 100,
        ServerOptions.STOP_VALUE       : 0.99,
        ServerOptions.PATIENT          : 10,
        ServerOptions.TARGET_METRIC    : MetricType.F1,
        ServerOptions.CLIENT_ACCEPTOR  : ClientAcceptorType.ALL_CLIENTS,
        ServerOptions.CLIENT_SELECTOR  : ClientSelectorType.ALL_CLIENTS,
        ServerOptions.MODEL_AGGREGATOR : AggregatorType.FED_AVG
    }

    # client_args = { }

    def topology():
        cleanup_fed_server_outputs()

        client_dimage = build_fed_node_docker_image("torch_client", "./fed/client_code/client_requirements.txt")["tag"]
        server_dimage = build_fed_node_docker_image("server", "./fed/server/server_requirements.txt")["tag"]

        net = MininetFed()
        try:
            s1 = net.addSwitch(name="s1", failMode='standalone')

            broker = net.addHost(name="broker", cls=FedBrokerNode)
            net.addLink(s1, broker)

            server = net.addHost(
                name="server",
                cls=FedServerNode,
                script="ehms_server.py",
                server_dimage=server_dimage,
                server_folder="./fed/server/",
                server_args=server_args
            )
            net.addLink(s1, server)

            for i in range(args.n_clients):
                cpath = clients_dir / f"client{i}"
                if not cpath.exists():
                    raise FileNotFoundError(f"Cliente não encontrado: {cpath}")

                # client_args é serializado em JSON pelo MininetFed -> chaves string e valores simples
                client_args = {
                    "attack_rate": args.attack_rate if i == args.malicious_client else 0.0,
                    "attack_seed": args.attack_seed,
                }
                print(f"[ATTACK] client{i}: {client_args}")

                # IMPORTANTE:
                # client_folder aponta para a pasta onde estão data_train.csv/data_test.csv + trainer code
                net.addHost(
                    name=f"client{i}",
                    cls=FedClientNode,
                    script="ehms_trainer.py",
                    dimage=client_dimage,
                    client_folder=str(cpath),
                    # passar client_args
                    client_args=client_args
                )
                net.addLink(s1, net.get(f"client{i}"))

            print(f'*** Starting network (variant={args.variant_tag}, attack_rate={args.attack_rate}, malicious_client={args.malicious_client})...\n')
            net.build()
            net.addNAT(name='nat0', linkTo='s1', ip='192.168.210.254').configDefault()
            s1.start([])
            net.runFed()
        finally:
            net.stop()

    # roda FL
    topology()

    # Coleta resultados do MininetFed
    # (assumindo que o server escreve em fed/server/)
    src_best = Path("fed/server/best.model.npz").resolve()
    src_metrics = Path("fed/server/metrics_summary.txt").resolve()

    if not src_best.exists():
        raise FileNotFoundError(f"Não encontrei: {src_best}")
    if not src_metrics.exists():
        raise FileNotFoundError(f"Não encontrei: {src_metrics}")

    shutil.copy2(src_best, out_dir / "best.model.npz")
    shutil.copy2(src_metrics, out_dir / "metrics_summary.txt")

    print(f"\n[OK] Copiado best.model.npz -> {out_dir / 'best.model.npz'}")
    print(f"[OK] Copiado metrics_summary.txt -> {out_dir / 'metrics_summary.txt'}")
    print(f"[OK] Resultados do fold/variant em: {out_dir}")


if __name__ == "__main__":
    main()
