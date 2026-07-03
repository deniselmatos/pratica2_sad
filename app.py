import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

st.set_page_config(
    page_title="Predição de Heating Load",
    layout="wide"
)

st.title("Predição da Carga de Aquecimento (Heating Load)")

st.write("""
Dashboard desenvolvido utilizando **Regressão Linear** para prever
a variável **Y1 (Heating Load)** a partir das características dos edifícios.
""")

@st.cache_data
def carregar_dados():
    return pd.read_excel("ENB2012_data.xlsx")

dados = carregar_dados()

X = dados.drop(columns=["Y1", "Y2"])
y = dados["Y1"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

predicoes = modelo.predict(X_test)

mae = mean_absolute_error(y_test, predicoes)
rmse = np.sqrt(mean_squared_error(y_test, predicoes))
r2 = r2_score(y_test, predicoes)

st.header("Avaliação do Modelo")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R²", f"{r2:.3f}")

st.info(
    f"O modelo explica aproximadamente **{r2*100:.1f}%** da variação do Heating Load."
)

st.header("Análise do Modelo")

col1, col2 = st.columns(2)

with col1:

    coeficientes = pd.DataFrame({
        "Variável": [
            "X1",
            "X2",
            "X3",
            "X4",
            "X5",
            "X6",
            "X7",
            "X8"
        ],
        "Coeficiente": modelo.coef_
    })

    fig = px.bar(
        coeficientes,
        x="Variável",
        y="Coeficiente",
        color="Coeficiente",
        title="Coeficientes da Regressão Linear"
    )

    st.plotly_chart(fig, width="stretch")

with col2:

    comparacao = pd.DataFrame({
        "Real": y_test,
        "Previsto": predicoes
    })

    fig = px.scatter(
        comparacao,
        x="Real",
        y="Previsto",
        trendline="ols",
        title="Valores Reais x Previstos"
    )

    st.plotly_chart(fig, width="stretch")

st.header("Simulador de Previsão")

c1, c2 = st.columns(2)

with c1:

    x1 = st.number_input(
        "Relative Compactness (X1)",
        value=float(dados["X1"].mean())
    )

    x2 = st.number_input(
        "Surface Area (X2)",
        value=float(dados["X2"].mean())
    )

    x3 = st.number_input(
        "Wall Area (X3)",
        value=float(dados["X3"].mean())
    )

    x4 = st.number_input(
        "Roof Area (X4)",
        value=float(dados["X4"].mean())
    )

with c2:

    x5 = st.selectbox(
        "Overall Height (X5)",
        sorted(dados["X5"].unique())
    )

    x6 = st.selectbox(
        "Orientation (X6)",
        sorted(dados["X6"].unique())
    )

    x7 = st.selectbox(
        "Glazing Area (X7)",
        sorted(dados["X7"].unique())
    )

    x8 = st.selectbox(
        "Glazing Area Distribution (X8)",
        sorted(dados["X8"].unique())
    )

if st.button("Prever Heating Load"):

    entrada = pd.DataFrame({
        "X1": [x1],
        "X2": [x2],
        "X3": [x3],
        "X4": [x4],
        "X5": [x5],
        "X6": [x6],
        "X7": [x7],
        "X8": [x8]
    })

    previsao = modelo.predict(entrada)

    st.success(
        f"### Heating Load previsto: **{previsao[0]:.2f}**"
    )
