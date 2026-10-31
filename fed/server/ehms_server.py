import os
import time

import numpy as np
from mininetfed.core.dto.client_state import ClientState
from mininetfed.core.dto.metrics import Metrics
from mininetfed.core.dto.training_data import TrainingData
from mininetfed.core.nodes.fed_server import FedServer
from numpy import ndarray


class EHMSServer(FedServer):
    def configure(self, server_id, broker_addr, server_folder, server_args):
        # O broker pode ainda não estar aceitando conexões quando o servidor sobe.
        # Tenta de novo por até ~60 s em vez de morrer com ConnectionRefusedError.
        for attempt in range(1, 31):
            try:
                return super().configure(server_id, broker_addr, server_folder, server_args)
            except OSError as e:
                print(f"[SERVER] broker indisponível ({e}); tentativa {attempt}/30, nova tentativa em 2 s")
                time.sleep(2)
        raise RuntimeError("[SERVER] não consegui conectar ao broker")

    def aggregate_model(self, training_responses : list[TrainingData], clients_state : dict[str, ClientState]) -> list[ndarray]:
        agg_model = super().aggregate_model(training_responses, clients_state)
        #self.save_round_checkpoint(agg_model)
        return agg_model

    def aggregate_metrics(self, clients_metrics : list[Metrics], n_samples : list[int]) -> Metrics:
        agg_metrics = super().aggregate_metrics(clients_metrics, n_samples)
        path = os.path.join(super().get_node_folder(), f"metrics_summary_{self.current_round}.txt")
        #agg_metrics.save_summary(path)
        return agg_metrics

    def save_round_checkpoint(self, agg_weights: list[ndarray]):
        """
        Salva pesos agregados do modelo global (lista de ndarrays) como NPZ.
        """
        path = os.path.join(super().get_node_folder(), f"model_{self.current_round}.npz")
        payload = {f"p{i}": w for i, w in enumerate(agg_weights)}
        np.savez_compressed(path, **payload)
        self.logger.info(f"[CHECKPOINT] Saved: {path}")
        print(f"[SERVER] Checkpoint salvo: {path}")