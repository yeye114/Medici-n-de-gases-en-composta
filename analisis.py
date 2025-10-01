import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from scipy.stats import gaussian_kde
import numpy as np

def ejecutar(df: pd.DataFrame, x_col: list, y_col: str, modelo_tipo: str = "linear"):
    try:
        # Separar variables predictoras y objetivo
        X = df[x_col].values
        y = df[y_col].values

        # Selección del modelo
        if modelo_tipo == "linear":
            modelo = LinearRegression()
        elif modelo_tipo == "ridge":
            modelo = Ridge(alpha=1.0)
        elif modelo_tipo == "lasso":
            modelo = Lasso(alpha=0.1)
        elif modelo_tipo == "elastic":
            modelo = ElasticNet(alpha=0.1, l1_ratio=0.5)
        else:
            st.error("⚠️ Modelo no reconocido. Usa: 'linear', 'ridge', 'lasso' o 'elastic'.")
            return

        # Entrenar el modelo
        modelo.fit(X, y)

        # Resultados numéricos
        st.subheader(f"Resultados de la Regresión ({modelo_tipo.capitalize()})")
        st.write("Coeficientes (pendientes):", modelo.coef_)
        st.write("Intercepto (ordenada al origen):", modelo.intercept_)

        r2 = modelo.score(X, y)
        st.write("R² (bondad de ajuste):", r2)

        # Predicciones
        y_pred = modelo.predict(X)
        df_resultados = pd.DataFrame({"Real": y, "Predicción": y_pred})
        st.write("Comparación entre valores reales y predichos:")
        st.dataframe(df_resultados)

        # Gráfico conjunto de distribución tipo campana
        st.subheader("Distribución del metano: Real vs Predicho")
        fig, ax = plt.subplots()

        # KDE para valores reales
        kde_real = gaussian_kde(y)
        x_vals = np.linspace(min(y.min(), y_pred.min()), max(y.max(), y_pred.max()), 200)
        ax.plot(x_vals, kde_real(x_vals), label="Real", color="blue", linewidth=2)

        # KDE para valores predichos
        kde_pred = gaussian_kde(y_pred)
        ax.plot(x_vals, kde_pred(x_vals), label="Predicción", color="red", linestyle="--", linewidth=2)

        # Configuración de la gráfica
        ax.set_title("Curvas de densidad (campana) del metano")
        ax.set_xlabel("Metano")
        ax.set_ylabel("Densidad")
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error al ejecutar la regresión: {e}")