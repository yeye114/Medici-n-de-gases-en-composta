import streamlit as st
from visualizar import mostrar as mostrar_visualizar
from analisis import mostrar as mostrar_analisis
from teoria import mostrar as mostrar_teoria

st.set_page_config(page_title="Menú Lateral", layout="wide")

st.sidebar.title("Menú de Navegación")
opcion = st.sidebar.selectbox(
    "Selecciona una opción:",
    ("Visualizar información", "Análisis", "Información teórica")
)

if opcion == "Visualizar información":
    mostrar_visualizar()
elif opcion == "Análisis":
    mostrar_analisis()
elif opcion == "Información teórica":
    mostrar_teoria()
