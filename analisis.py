import streamlit as st
import numpy as np
import pandas as pd

def mostrar():
    st.header("Análisis de datos")
    st.write("Aquí puedes mostrar gráficos, estadísticas o modelos predictivos.")

    # Ejemplo: gráfico de líneas con datos aleatorios
    data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Sensor A', 'Sensor B', 'Sensor C']
    )
    st.line_chart(data)

    # Estadísticas básicas
    st.write("Resumen estadístico:")
    st.write(data.describe())
