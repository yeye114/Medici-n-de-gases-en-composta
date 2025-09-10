import streamlit as st
import pandas as pd
import numpy as np

def mostrar():
    st.header("Visualización de información")
    st.write("Aquí puedes mostrar tablas, gráficos o datos relevantes.")

    # Ejemplo: mostrar un DataFrame
    df = pd.DataFrame(
        np.random.randn(10, 3),
        columns=['Sensor A', 'Sensor B', 'Sensor C']
    )
    st.dataframe(df)

    # Ejemplo: gráfico de barras
    st.bar_chart(df)
