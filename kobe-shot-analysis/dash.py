import joblib
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import os
from utils import load_best_model_from_last_run
from sklearn.metrics import classification_report, confusion_matrix


def load_data():
    path = "./data/02_intermediate/processed_prod_data.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        st.warning("Base de produção não encontrada.")
        return pd.DataFrame()


df = load_data()

st.set_page_config(page_title="Análise de Arremessos do Kobe", layout="wide")


@st.cache_resource
def load_best_model():
    return load_model("best_model")


model = load_best_model()
st.title("🏀 Dashboard - Monitoramento de Operação")

# Tabs
aba = st.tabs(["📊 Visão Geral do Modelo", "🎯 Simulador de Arremesso"])

# ======= Aba 1: Visão Geral ======= #
with aba[0]:
    st.subheader("🔎 Métricas de Desempenho do Modelo")

    if not df.empty and "shot_made_flag" in df.columns:
        X = df.drop(columns=["shot_made_flag"])
        y_true = df["shot_made_flag"]
        y_pred = model.predict(X)

        col1, col2, col3 = st.columns(3)
        report = classification_report(y_true, y_pred, output_dict=True)

        print(report)

        col1.metric("🎯 Acurácia", f"{report['accuracy']:.2%}")
        col2.metric("📦 Precision", f"{report['0.0']['precision']:.2%}")
        col3.metric("💥 F1-Score", f"{report['0.0']['f1-score']:.2%}")

    st.info("Essas métricas refletem o desempenho atual do modelo em dados de teste.")

    st.subheader("Distribuição das Previsões")
    pred_data = pd.read_parquet("data/08_reporting/prediction_prod.parquet")
    st.write("Amostra das predições:", pred_data.head())

    if "Score" in pred_data.columns:
        st.subheader("Distribuição de Confiança das Previsões")
        fig_score, ax_score = plt.subplots()
        sns.histplot(pred_data["Score"], bins=20, kde=True, ax=ax_score)
        st.pyplot(fig_score)

    # # Gráfico: Distribuição de acertos e erros
    st.subheader("🎯 Distribuição de Acertos e Erros")
    fig_dist, ax2 = plt.subplots()
    df["shot_made_flag"].value_counts().plot(kind="bar", ax=ax2)
    ax2.set_xticklabels(["Erro", "Acerto"], rotation=0)
    ax2.set_ylabel("Quantidade")
    st.pyplot(fig_dist)

    st.subheader("🔗 Correlação entre Variáveis")
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax
    )
    st.pyplot(fig_corr)

    st.subheader("📌 Matriz de confusão")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # st.subheader("🎯 Distribuição das Predições")
    # df["prediction"] = y_pred
    # fig2 = px.histogram(df, x="prediction", color="prediction", barmode="group")
    # st.plotly_chart(fig2)

# ======= Aba 2: Inputs do Usuário ======= #
with aba[1]:
    st.subheader("📝 Preencha os dados do arremesso para prever o resultado")

    col1, col2, col3 = st.columns(3)
    with col1:
        action_type = st.selectbox(
            "Tipo de Ação", ["Jump Shot", "Layup Shot", "Dunk Shot"]
        )
        period = st.selectbox("Período do jogo", [1, 2, 3, 4])
        shot_distance = st.slider("Distância do Arremesso (ft)", 0, 40, 15)
    with col2:
        shot_type = st.selectbox(
            "Tipo de Arremesso", ["2PT Field Goal", "3PT Field Goal"]
        )
        minutes_remaining = st.slider("Minutos Restantes", 0, 12, 6)
        seconds_remaining = st.slider("Segundos Restantes", 0, 59, 30)
    with col3:
        opponent = st.selectbox("Adversário", ["LAL", "BOS", "CHI", "MIA", "NYK"])
        home = st.radio("Jogo em Casa?", ["Sim", "Não"])

    input_dict = {
        "action_type": action_type,
        "period": period,
        "shot_distance": shot_distance,
        "shot_type": shot_type,
        "minutes_remaining": minutes_remaining,
        "seconds_remaining": seconds_remaining,
        "opponent": opponent,
        "home": 1 if home == "Sim" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("🔮 Prever Arremesso"):
        output = predict_model(model, data=input_df)
        pred = output.loc[0, "Label"]
        prob = output.loc[0, "Score"]

        if pred == 1:
            st.success(f"✅ Kobe acertaria esse arremesso! (Confiança: {prob:.2%})")
        else:
            st.error(f"❌ Kobe erraria esse arremesso. (Confiança: {(1 - prob):.2%})")
