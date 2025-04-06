import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import os
from utils import load_best_model_from_last_run

st.set_page_config(layout="wide")


# =============================
# 📂 Load Data
# =============================
@st.cache_data
def load_data():
    path = "./data/02_intermediate/processed_prod_data.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        st.warning("Base de produção não encontrada.")
        return pd.DataFrame()


df = load_data()


model, last_run_id = load_best_model_from_last_run()

# =============================
# 🎯 Run predictions
# =============================
if not df.empty and "shot_made_flag" in df.columns:
    X = df.drop(columns=["shot_made_flag"])
    y_true = df["shot_made_flag"]
    y_pred = model.predict(X)

    st.title("🏀 Dashboard - Monitoramento de Operação")

    # =============================
    # 📊 Métricas
    # =============================
    col1, col2, col3 = st.columns(3)
    report = classification_report(y_true, y_pred, output_dict=True)

    print(report)

    col1.metric("🎯 Acurácia", f"{report['accuracy']:.2%}")
    col2.metric("📦 Precision", f"{report['0.0']['precision']:.2%}")
    col3.metric("💥 F1-Score", f"{report['0.0']['f1-score']:.2%}")

    # =============================
    # 📈 Gráficos
    # =============================
    st.subheader("📌 Matriz de confusão")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("🎯 Distribuição das Predições")
    df["prediction"] = y_pred
    fig2 = px.histogram(df, x="prediction", color="prediction", barmode="group")
    st.plotly_chart(fig2)

    # =============================
    # 🔍 Últimas previsões
    # =============================
    st.subheader("🕵️ Últimas 10 Previsões")
    st.dataframe(
        df[["prediction"] + [col for col in df.columns if col != "prediction"]].tail(10)
    )

else:
    st.warning("Base de produção vazia ou sem coluna 'shot_made_flag'.")
