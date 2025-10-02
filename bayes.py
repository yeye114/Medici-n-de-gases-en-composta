import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def ejecutar(df: pd.DataFrame, x_col: list, y_col: str):
    try:
        # Separar variables predictoras y objetivo
        X = df[x_col].values
        y = df[y_col].values

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Crear y entrenar el modelo Bayesiano
        model = BayesianRidge()
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)

        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Resultados del Modelo Bayesiano (Bayesian Ridge)")
        st.write(f"**Error cuadrático medio (MSE):** {mse:.4f}")
        st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")

        # Mostrar coeficientes del modelo
        st.subheader("Coeficientes del modelo")
        coef_df = pd.DataFrame({
            "Variable": x_col,
            "Coeficiente": model.coef_
        })
        st.dataframe(coef_df)

        # Comparación en tabla
        st.subheader("Comparación entre valores reales y predichos")
        comparacion_df = pd.DataFrame({
            "Valor Real": y_test,
            "Valor Predicho": y_pred
        })
        st.dataframe(comparacion_df)

        # Gráfico: valores reales vs predichos
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("Valores Reales de Metano")
        ax.set_ylabel("Valores Predichos de Metano")
        ax.set_title("Comparación entre valores reales y predichos")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error al ejecutar el modelo Bayesiano: {e}")
