import os
import numpy as np
from numpy import ndarray

from mininetfed.core.dto.client_info import ClientInfo
from mininetfed.core.dto.dataset_info import DatasetInfo
from mininetfed.core.dto.metrics import Metrics, MetricType
from mininetfed.core.nodes.fed_client import FedClient

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================
# Hiperparâmetros locais
# ============================

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
N_EPOCHS = 10   # nº de épocas de treino local por round

TRAIN_FILE = "data_train.csv"
TEST_FILE  = "data_test.csv"

LABEL_COL = "Label"
ID_COL = "instance_id"  # pode existir; não entra como feature


# ============================
# Dataset tabular para PyTorch
# ============================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # BCEWithLogitsLoss espera alvo float com shape (N, 1)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================
# MLP do artigo (EHMS ANN)
# ============================

class EHMSANN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(40, 20),
            nn.ReLU(),

            nn.Linear(20, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 1),  # saída: logit
        )

    def forward(self, x):
        return self.net(x)


# ============================
# Cliente Federado (PyTorch)
# ============================

class TrainerEHMS(FedClient):
    """
    Cliente federado usando ANN em PyTorch para o dataset EHMS.

    ATUALIZAÇÕES:
    - Agora carrega os dados a partir de:
        <path_to_data>/data_train.csv
        <path_to_data>/data_test.csv
      (pois o pré-processamento e o split já foram feitos pelo script gerador)
    - Não usa mais .npz nem train_test_split aqui.
    - Mantém padronização local com StandardScaler (fit no treino, transform no teste).
    """

    def __init__(self):
        super().__init__()
        self.model: EHMSANN | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.features: list[str] | None = None

        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.criterion = None
        self.optimizer = None

        # Implementar etodo configure do fedclient
        # chamar .super e obter o parametro que indica se o cliente e malicioso

        # Ataque de label flipping (configurado via client_args em configure())
        self.attack_rate: float = 0.0
        self.attack_seed: int = 42
        self.flip_idx:np.ndarray | None = None   # índices (do treino) com rótulo invertido
        self.poisoned_train_loader: DataLoader | None = None

    # -------------------------
    # 0) Configuração do cliente (lê client_args)
    # -------------------------
    def configure(self, client_id, broker_id, client_folder, client_args):
        # super().configure() conecta ao broker e já chama prepare_data(),
        # então X_train/y_train estão disponíveis depois desta linha
        super().configure(client_id, broker_id, client_folder, client_args)

        args = client_args or {}
        self.attack_rate = float(args.get("attack_rate", 0.0))
        self.attack_seed = int(args.get("attack_seed", 42))

        if not 0.0 <= self.attack_rate <= 1.0:
            raise ValueError(f"[CLIENT {client_id}] attack_rate inválido: {self.attack_rate}")
        
        if self.attack_rate > 0.0:
            # Sorteia UMA vez quais amostras terão o rótulo invertido; o mesmo subconjunto é usado
            # em todas as épocas e rounds. Com a mesma semente, a permutação é a mesma para 30/60/100%,
            # então o subconjunto de 30% está contido no de 60%.
            n = len(self.y_train)
            n_flip = int(round(self.attack_rate * n))
            rng = np.random.default_rng(self.attack_seed)
            self.flip_idx = np.sort(rng.permutation(n)[:n_flip])
            print(f"[CLIENT {client_id}] MALICIOSO: label flipping em {n_flip}/{n} amostras ({self.attack_rate:.0%}), seed={self.attack_seed}")
        else:
            print(f"[CLIENT {client_id}] Benigno (sem ataque).")

    def _get_poisoned_train_loader(self) -> DataLoader:
        """Monta (uma única vez) o DataLoader de treino com os rótulos invertidos"""
        if self.poisoned_train_loader is None:
            y_poisoned = self.y_train.copy()
            y_poisoned[self.flip_idx] = 1 - y_poisoned[self.flip_idx]

            n_changed = int((y_poisoned != self.y_train).sum())
            print(f"[CLIENT {self.get_client_id()}] Rótulos invertidos: {n_changed} | classe 1 antes={int((self.y_train == 1).sum())} depois={int((y_poisoned == 1).sum())}")

            poisoned_ds = TabularDataset(self.X_train, y_poisoned)
            self.poisoned_train_loader = DataLoader(poisoned_ds, batch_size=BATCH_SIZE, shuffle=True)

            # O cliente malicioso segue o pipeline ormal sobre os dados envenenados:
            # pos_weight recalculado com os rótulos que ele de fato usa no treino.
            n_pos = int((y_poisoned == 1).sum())
            n_neg = int((y_poisoned == 0).sum())
            pos_weight_value = n_neg / max(n_pos, 1)
            pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"[CLIENT {self.get_client_id()}] pos_weight (dados envenenados): {pos_weight_value:.4f}")

        return self.poisoned_train_loader

    # -------------------------
    # 1) Preparar dados locais
    # -------------------------
    def prepare_data(self, path_to_data: str) -> DatasetInfo:
        """
        path_to_data: diretório do cliente (ex.: fed/clients/client0)
        Espera encontrar:
          - data_train.csv
          - data_test.csv
        """
        train_path = os.path.join(path_to_data, TRAIN_FILE)
        test_path = os.path.join(path_to_data, TEST_FILE)

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {test_path}")

        print(f"[CLIENT {self.get_client_id()}] Lendo treino: {train_path}")
        df_train = np.loadtxt(train_path, delimiter=",", skiprows=1)  # fallback rápido? NÃO (perde header)
        # -> Vamos usar pandas, pois precisamos do header para selecionar features corretamente:
        import pandas as pd
        df_train = pd.read_csv(train_path)

        print(f"[CLIENT {self.get_client_id()}] Lendo teste : {test_path}")
        df_test = pd.read_csv(test_path)

        if LABEL_COL not in df_train.columns or LABEL_COL not in df_test.columns:
            raise RuntimeError(f"[CLIENT {self.get_client_id()}] CSVs devem conter coluna '{LABEL_COL}'.")

        excluded = {LABEL_COL, ID_COL}
        features = [c for c in df_train.columns if c not in excluded]
        features_test = [c for c in df_test.columns if c not in excluded]

        if features_test != features:
            raise RuntimeError(
                f"[CLIENT {self.get_client_id()}] Features de treino e teste não batem.\n"
                f"Treino ({len(features)}): {features}\n"
                f"Teste  ({len(features_test)}): {features_test}"
            )

        self.features = features

        X_train = df_train[features].values.astype("float32")
        y_train = df_train[LABEL_COL].values.astype("int64")

        X_test = df_test[features].values.astype("float32")
        y_test = df_test[LABEL_COL].values.astype("int64")

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        # Datasets e loaders
        train_ds = TabularDataset(self.X_train, self.y_train)
        test_ds = TabularDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Modelo + loss + otimizador
        input_dim = self.X_train.shape[1]
        self.model = EHMSANN(input_dim=input_dim).to(self.device)

        # Classe positiva = 1, negativa = 0
        n_pos = int((self.y_train == 1).sum())
        n_neg = int((self.y_train == 0).sum())
        pos_weight_value = n_neg / max(n_pos, 1)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device)

        print(f"[CLIENT {self.get_client_id()}] Classe positiva (1): {n_pos}")
        print(f"[CLIENT {self.get_client_id()}] Classe negativa (0): {n_neg}")
        print(f"[CLIENT {self.get_client_id()}] pos_weight usado: {pos_weight_value:.4f}")

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        return DatasetInfo(client_id=self.get_client_id(), num_samples=int(self.X_train.shape[0]))

    # -------------------------
    # 2) Informações do cliente
    # -------------------------
    def set_client_info(self, client_info: ClientInfo):
        # mantém compatibilidade com seu fluxo
        return ClientInfo(self.get_client_id())

    # -------------------------
    # 3) Treino local
    # -------------------------
    def fit(self) -> bool:
        if self.model is None or self.train_loader is None:
            print(f"[CLIENT {self.get_client_id()}] Model ou train_loader não inicializados.")
            return False

        try:
            # Se malicioso, flip de labels: treina sobre o subconjunto com rótulos invertidos.
            # Os dados de teste (evaluate) não são alterados.
            if self.attack_rate > 0.0:
                train_loader = self._get_poisoned_train_loader()
            else:
                train_loader = self.train_loader

            self.model.train()
            for epoch in range(1, N_EPOCHS + 1):
                running_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    self.optimizer.zero_grad()
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * X_batch.size(0)

                avg_loss = running_loss / len(train_loader.dataset)
                print(f"[CLIENT {self.get_client_id()}] Epoch {epoch}/{N_EPOCHS} - loss={avg_loss:.4f}")

            return True
        except Exception as e:
            print(f"[CLIENT {self.get_client_id()}] Training failed: {e}")
            return False

    # -------------------------
    # 4) Avaliação local
    # -------------------------
    def evaluate(self) -> Metrics:
        if self.model is None or self.test_loader is None:
            print(f"[CLIENT {self.get_client_id()}] Model ou test_loader não inicializados.")
            return Metrics(client_id=self.get_client_id(), metrics={})

        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)

                logits = self.model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.append(probs)
                all_labels.append(y_batch.cpu().numpy())

        all_probs = np.vstack(all_probs).ravel()
        all_labels = np.vstack(all_labels).ravel().astype("int32")

        preds = (all_probs >= 0.5).astype("int32")

        cm = confusion_matrix(all_labels, preds, labels=[0, 1])
        acc = float((preds == all_labels).mean())

        metrics = {
            MetricType.CONFUSION_MATRIX: cm.tolist(),
            MetricType.ACCURACY: acc,  # opcional, mas útil
        }

        return Metrics(client_id=self.get_client_id(), metrics=metrics)

    # -------------------------
    # 5) Sincronizar pesos com o servidor
    # -------------------------
    def update_weights(self, global_weights: list[ndarray]):
        """
        Atualiza pesos do modelo local com os pesos globais.
        """
        if self.model is None:
            if self.X_train is not None:
                input_dim = self.X_train.shape[1]
                self.model = EHMSANN(input_dim=input_dim).to(self.device)
            else:
                raise RuntimeError("Model ainda não inicializado e não há dados locais para inferir input_dim.")

        with torch.no_grad():
            for param, w in zip(self.model.parameters(), global_weights):
                w_t = torch.from_numpy(w).to(self.device)
                if param.data.shape != w_t.shape:
                    raise RuntimeError(
                        f"[CLIENT {self.get_client_id()}] Shape mismatch ao atualizar pesos: "
                        f"param={tuple(param.data.shape)} vs w={tuple(w_t.shape)}"
                    )
                param.data.copy_(w_t)

    def get_weights(self) -> list[ndarray]:
        """
        Retorna pesos locais como lista de ndarrays para agregação.
        """
        if self.model is None:
            raise RuntimeError("Model ainda não inicializado em get_weights().")

        return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]
