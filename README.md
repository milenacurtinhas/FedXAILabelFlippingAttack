# Do Aprendizado Centralizado ao Federado: O Que Acontece com a Explicabilidade dos Modelos?

Este repositório reúne o código referente ao artigo *Do Aprendizado Centralizado ao Federado: 
O Que Acontece com a Explicabilidade dos Modelos?*. O objetivo do trabalho foi propor uma 
metodologia para analisar o impacto do **aprendizado federado (FL)** sobre explicações **SHAP** 
em diferentes cenários de treino e particionamento de dados. O dataset EHMS 
([Enhanced Healthcare Monitoring System](https://www.cse.wustl.edu/~jain/ehms/index.html)).
foi usado para aplicar a metodologia proposta.

## Visão geral do fluxo

A aplicação da metodologia consiste nos seguintes passos:

1. criação os folds para treinamento usando a estratégia de validação cruzada;
2. divisão do dataset para geração dos clientes para o treinamento federado, com particionamento seguindo a mesma distribuição do dataset original (Uni) e distribuição desbalanceada em relação ao dataset original (NonUni); 
3. treinamentos dos modelos central, federado Uni e federado NonUni, com validação cruzada, em cada cenário de ataque por *label flipping*;
4. calculo dos valores SHAP para os modelos e cenários de comparação;
5. comparação entre as explicações para os cenários;
6. agregação de resultados globais;
7. geração de gráficos para visualização das métricas de comparação entre os diferentes cenários.

## Cenários de ataque por *label flipping*

Além do fluxo original, o treinamento federado pode ser executado com **um cliente malicioso**
que inverte os rótulos (0 ↔ 1) de parte dos seus dados de treino. O cliente malicioso é sempre o
mesmo em todos os cenários (padrão: `client0`) e os demais clientes são benignos.

| Cenário | `--attack-rate` | Descrição | Sufixo usado nas pastas |
|---|---|---|---|
| 1 | `0.0` | código original, todos os clientes benignos, sem ataque | `lf00` |
| 2 | `0.3` | 30% dos dados de treino do cliente malicioso com rótulo invertido (escolha aleatória) | `lf30` |
| 3 | `0.6` | 60% dos dados de treino do cliente malicioso com rótulo invertido (escolha aleatória) | `lf60` |
| 4 | `1.0` | 100% dos dados de treino do cliente malicioso com rótulo invertido | `lf100` |

Como funciona:

- `fed/ehms_fed.py` recebe `--attack-rate`, `--malicious-client` (padrão `0`) e `--attack-seed`
  (padrão `42`) e repassa a cada cliente um `client_args` com a taxa de ataque
  (`0.0` para os clientes benignos).
- `fed/client_code/ehms_trainer.py` lê esses parâmetros em `configure()`. As amostras a inverter
  são sorteadas **uma única vez** e usadas em todas as épocas e rounds. Com a mesma semente, as
  amostras invertidas no cenário de 30% estão contidas nas do cenário de 60%.
- Apenas os dados de **treino** do cliente malicioso são alterados; os dados de teste não mudam.
- O modelo **central não é afetado** pelo ataque, por isso é treinado apenas no cenário 1 e
  reaproveitado nos demais.
- `run_train_all_folds.py` repassa `--attack-rate`, `--malicious-client` e `--attack-seed` para
  `fed/ehms_fed.py`.

Cada cenário grava seus resultados em pastas próprias (`results_lf30`, `shap_lf30`,
`comparisons_lf30`, ...), para que um cenário não sobrescreva o outro.

## Pré-requisitos

### Sistema

- Ubuntu versões 20.04, 22.04 ou 24.04
- Python 3
- MininetFed 2.0 instalado e funcional

**Para instalar o MininetFed 2.0, siga as instruções em:**

https://github.com/lprm-ufes/MininetFed-2.0-SBRC-2026

### Configuração do Ambiente Python


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn torch shap matplotlib scipy joblib
```

## Sequência de comandos

### 1) Clonar o repositório

```bash
git clone https://github.com/danielrt/fed_xai_sbrc2026.git
cd fed_xai_sbrc2026
```

---

### 2) Criar os folds centrais, os datasets federados e os diretórios montados dos clientes

```bash
python3 run_cv_folds_pipeline.py \
  --csv dataset/wustl-ehms-2020_with_attacks_categories.csv \
  --cv-root cv \
  --k 5 \
  --seed 42 \
  --n-clients 4 \
  --alpha-size 0.8 \
  --alpha-class 0.8 \
  --variants iid,noniid \
  --client-code-dir fed/client_code \
  --mount-root mounted_clients \
  --clean
```

> **Importante:** este passo copia o conteúdo de `fed/client_code/` (incluindo `ehms_trainer.py`)
> para cada cliente em `mounted_clients/`. Se `ehms_trainer.py` for alterado, este passo deve ser
> executado novamente (com `--clean`). Ele é executado **uma única vez** para os 4 cenários, pois o
> ataque é aplicado em tempo de treino, e não nos arquivos de dados.

### 3) Treinar os modelos central, federado IID e federado non-IID (por cenário)

**Cenário 1 (sem ataque):** treina o modelo central e os federados IID e non-IID.

```bash
python3 run_train_all_folds.py \
  --k 5 \
  --cv-root cv \
  --mount-root mounted_clients \
  --results-root results_lf00 \
  --n-clients 4 \
  --central-script ehms_central.py \
  --fed-script fed/ehms_fed.py \
  --fed-server-dir fed/server \
  --seed 42 \
  --epochs 150 \
  --patience 10 \
  --attack-rate 0.0
```

**Cenários 2, 3 e 4 (com ataque):** treina apenas os modelos federados e copia o modelo central
do cenário 1, pois os passos seguintes exigem `k_foldX/central/best_model.npz` em cada pasta de
resultados.

```bash
for CEN in 30 60 100; do
  RATE=$(python3 -c "print(${CEN}/100)")

  python3 run_train_all_folds.py \
    --k 5 \
    --cv-root cv \
    --mount-root mounted_clients \
    --results-root results_lf${CEN} \
    --n-clients 4 \
    --skip-central \
    --fed-script fed/ehms_fed.py \
    --fed-server-dir fed/server \
    --seed 42 \
    --attack-rate ${RATE} \
    --malicious-client 0 \
    --attack-seed 42

  for i in 0 1 2 3 4; do
    cp -r results_lf00/k_fold${i}/central results_lf${CEN}/k_fold${i}/
  done
done
```

> Para rodar um cenário isolado, basta usar o comando de dentro do laço com `CEN` e `RATE`
> definidos manualmente (por exemplo, `CEN=60` e `RATE=0.6`).

### Passos 4 a 7: executar para cada cenário

Os passos a seguir são executados **uma vez por cenário**. Os comandos estão escritos com a
variável `CEN`; defina-a antes de rodar cada bloco, ou use o laço abaixo, que executa os passos 4,
5 e 6 para todos os cenários:

```bash
for CEN in 00 30 60 100; do
  python3 run_shap_all_folds.py --k 5 --cv-root cv --results-root results_lf${CEN} \
    --out-root shap_lf${CEN} --compute-script compute_shap_values.py \
    --background-size 500 --label-col Label --id-col instance_id --n-clients 4

  python3 run_compare_shap_all_folds.py --k 5 --shap-root shap_lf${CEN} \
    --out-root comparisons_lf${CEN} --compare-script compare_shap_instancewise.py \
    --rbo-p 0.9 --topk 5,10,15

  python3 summarize_shap_global_means.py --shap-root shap_lf${CEN} \
    --out-root global_means_lf${CEN} --ci-level 0.95
done
```

> O passo 5 compara explicações **dentro** de um mesmo cenário (por exemplo, central × federado
> IID). A comparação **entre** cenários de ataque, usada nos gráficos ECDF e PMF, é feita no passo
> 5b, depois que os SHAP dos 4 cenários tiverem sido calculados.

### 4) Calcular os valores SHAP para todos os folds e cenários

```bash
python3 run_shap_all_folds.py \
  --k 5 \
  --cv-root cv \
  --results-root results_lf${CEN} \
  --out-root shap_lf${CEN} \
  --compute-script compute_shap_values.py \
  --background-size 500 \
  --label-col Label \
  --id-col instance_id \
  --n-clients 4
```

### 5) Comparar explicações SHAP entre cenários

```bash
python3 run_compare_shap_all_folds.py \
  --k 5 \
  --shap-root shap_lf${CEN} \
  --out-root comparisons_lf${CEN} \
  --compare-script compare_shap_instancewise.py \
  --rbo-p 0.9 \
  --topk 5,10,15
```

### 5b) Comparar explicações SHAP entre cenários de ataque (MF_DL)

Executado **uma única vez**, depois do passo 4 dos 4 cenários. Compara, em cada fold, os SHAP do
contexto MF_DL (modelo federado aplicado aos dados de teste locais dos clientes, juntando os 4
clientes) do cenário sem ataque com cada cenário com ataque:

- MF_DL (Uni): 0% × 30%, 0% × 60%, 0% × 100%
- MF_DL (NonUni): 0% × 30%, 0% × 60%, 0% × 100%

```bash
python3 run_compare_attack_scenarios.py \
  --k 5 \
  --shap-prefix shap_lf \
  --baseline 00 \
  --attacks 30,60,100 \
  --out-root comparisons_attack \
  --compare-script compare_shap_instancewise.py \
  --rbo-p 0.9 \
  --topk 5,10,15
```

Lê `shap_lf00/`, `shap_lf30/`, `shap_lf60/` e `shap_lf100/` e grava em
`comparisons_attack/MF_DL_<Uni|NonUni>_lf00_vs_lf<30|60|100>/fold<i>/`. As instâncias batem entre
os cenários porque os dados de teste dos clientes são os mesmos: o ataque altera apenas os rótulos
de treino do cliente malicioso.

### 6) Agregar as médias globais dos SHAP entre folds

```bash
python3 summarize_shap_global_means.py \
  --shap-root shap_lf${CEN} \
  --out-root global_means_lf${CEN} \
  --ci-level 0.95
```

### 7) Gerar gráficos para visualização das métricas de comparação entre os diferentes cenários.

Defina o cenário antes de gerar os gráficos (repita para `00`, `30`, `60` e `100`):

```bash
CEN=30
```

```bash
python3 plot_top15_shap_global_bars.py \
  --means-root global_means_lf${CEN} \
  --out-dir fig_top_features_ci_lf${CEN} \
  --top-n 15
```

**O que gera:**

Arquivos `.pdf` e `.png` em `fig_top_features_ci_lf${CEN}/`, por exemplo:

- `top15_MC_DC_global_means.pdf`
- `top15_MA_DC_iid_global_means.pdf`
- `top15_MA_DC_noniid_global_means.pdf`
- e equivalentes para os cenários `MA_DL_*` disponíveis

Os gráficos ECDF e PMF abaixo comparam os **cenários de ataque** e são gerados **uma única vez**, a
partir da saída do passo 5b (não dependem de `CEN`):

```bash
python3 plot_ecdf_cosine_distance.py \
  --compare-root comparisons_attack \
  --k 5 \
  --out-dir fig_ecdf_cosine_attack
```

**O que gera:** 6 figuras em `fig_ecdf_cosine_attack/`, uma curva por fold em cada:

- `ecdf_cosine_MF_DL_Uni_lf00_vs_lf30`, `..._lf00_vs_lf60`, `..._lf00_vs_lf100`
- `ecdf_cosine_MF_DL_NonUni_lf00_vs_lf30`, `..._lf00_vs_lf60`, `..._lf00_vs_lf100`

```bash
python3 plot_pmf_jaccard_topk_informative_bins.py \
  --compare-root comparisons_attack \
  --kfolds 5 \
  --out-dir fig_pmf_jaccard_attack \
  --threshold 0.6
```

**O que gera:** em `fig_pmf_jaccard_attack/`, uma figura por contexto e por k (top-5, top-10 e
top-15), com 3 barras por bin (0% × 30%, 0% × 60%, 0% × 100%):

- `pmf_matches_informative_MF_DL_Uni_top5`, `..._top10`, `..._top15`
- `pmf_matches_informative_MF_DL_NonUni_top5`, `..._top10`, `..._top15`

```bash
python3 plot_pmf_topk_sign_consistency_informative_bins.py \
  --compare-root comparisons_attack \
  --kfolds 5 \
  --out-dir fig_pmf_sign_attack \
  --threshold 0.6
```

**O que gera:** em `fig_pmf_sign_attack/`, a mesma organização do Jaccard:

- `pmf_sign_informative_MF_DL_Uni_top5`, `..._top10`, `..._top15`
- `pmf_sign_informative_MF_DL_NonUni_top5`, `..._top10`, `..._top15`

Para gerar só um contexto ou só alguns ataques, os três scripts aceitam `--contexts Uni` (ou
`NonUni`) e `--attacks 30,100`.
