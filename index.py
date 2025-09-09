import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Menú Lateral", layout="wide")

# Título principal
st.title("Medición de Gases en Composta Bovino-Ovino ")

# Menú lateral
st.sidebar.title("Menú de Navegación")
opcion = st.sidebar.selectbox(
    "Selecciona una opción:",
    ("Visualizar información", "Análisis", "Información teórica")
)

# Contenido dinámico según la opción seleccionada
if opcion == "Visualizar información":
    st.header("Visualización de información")
    st.write("Aquí puedes mostrar tablas, gráficos o datos relevantes.")
   

elif opcion == "Análisis":
    st.header("Análisis de datos")
    st.write("Aquí puedes mostrar gráficos, estadísticas o modelos predictivos.")
    
elif opcion == "Información teórica":
    st.header("Información teórica")
    

