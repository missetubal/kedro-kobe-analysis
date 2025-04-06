# 🏀 Kobe Shot Analysis

Este projeto utiliza ciência de dados e aprendizado de máquina para prever se Kobe Bryant acertou ou errou um arremesso com base em dados históricos de jogadas. A iniciativa faz parte de um pipeline de Machine Learning Engineering usando o framework **Kedro**, com monitoramento via **MLflow** e visualização com **Streamlit**.

## 🔍 Objetivo

Desenvolver e operacionalizar um modelo de machine learning para classificar arremessos de Kobe Bryant como “acerto” (`1`) ou “erro” (`0`), utilizando técnicas modernas de MLOps e engenharia de dados.

---

## 🧰 Tecnologias e Ferramentas

- **Python 3.11+**
- **Kedro** – Estruturação do pipeline de dados e experimentos.
- **MLflow** – Rastreamento, registro e gerenciamento de modelos.
- **Scikit-learn** – Modelos de machine learning.
- **Streamlit** – Dashboard de monitoramento e comparação de modelos.
- **Pandas / NumPy / Matplotlib / Seaborn** – Manipulação de dados e visualizações.

---

## 📂 Estrutura do Projeto

```
kobe-shot-analysis/
│
├── data/                        # Armazenamento de dados (raw, intermediate, processed, etc)
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 04_feature/
│   └── 05_model_input/
│
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Registro dos datasets
│   │   ├── parameters.yml       # Parâmetros globais
│   │   └── logging.yml          # Configurações de logging
│
├── kobe_shot_analysis/          # Código principal da pipeline
│   ├── nodes/
│   │   ├── data_engineering.py
│   │   ├── modeling.py
│   │   └── evaluation.py
│   ├── pipelines/
│   ├── hooks.py
│   └── settings.py
│
├── notebooks/                   # Notebooks para exploração e desenvolvimento
│
├── mlruns/                      # Diretório do MLflow
│
├── app.py                       # Script principal para previsão com o melhor modelo
├── dashboard.py                 # Dashboard Streamlit de monitoramento
├── requirements.txt             # Dependências do projeto
└── README.md                    # Este arquivo
```

---

## 🚀 Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/missetubal/kedro-kobe-analysis.git
cd kedro-kobe-analysis
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o pipeline Kedro

```bash
kedro run
```

### 5. Visualize os experimentos com MLflow

```bash
mlflow ui
```

Acesse [http://localhost:5000](http://localhost:5000) para visualizar os experimentos.

### 6. Execute o dashboard de monitoramento

```bash
streamlit run dashboard.py
```

---

## 📊 Resultados

Os modelos implementados até o momento incluem:

- Regressão Logística
- Árvore de Decisão

A avaliação de desempenho é feita com base em métricas como:

- Acurácia
- Precisão
- Recall
- F1-Score
- Matriz de Confusão
- ROC AUC

Os melhores modelos são registrados automaticamente no MLflow, e o `app.py` utiliza o modelo com melhor desempenho para previsão em novos dados.

---

## 📌 Próximos Passos

- Aprimoramento do feature engineering (clustering, encoding, etc).
- Inclusão de novos modelos como Random Forest, Gradient Boosting, XGBoost.
- Validação cruzada mais robusta e balanceamento de classes.
- Deploy completo em nuvem (AWS/GCP).

---

## 👨‍💻 Autor

Desenvolvido por [@missetubal](https://github.com/missetubal)  
