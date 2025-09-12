import streamlit as st
from forest import ejecutar as ejecutar_forest
from bayes import ejecutar as ejecutar_bayes

def mostrar():
    st.title("Análisis de Datos con Algoritmos")

    # Verificar si hay un archivo cargado
    if 'df' not in st.session_state:
        st.warning("Primero debes subir un archivo en 'Visualizar información'.")
        return

    df = st.session_state.df

    # Inicializar estado para algoritmo
    if 'algoritmo' not in st.session_state:
        st.session_state.algoritmo = None
    if 'x_col' not in st.session_state:
        st.session_state.x_col = []
    if 'y_col' not in st.session_state:
        st.session_state.y_col = None

    # Selección de algoritmo
    st.subheader("Selecciona el algoritmo")
    algoritmo = st.selectbox("Algoritmo", ["Random Forest", "Red de Bayes"])

    # Detectar cambio de algoritmo y limpiar variables
    if st.session_state.algoritmo != algoritmo:
        st.session_state.algoritmo = algoritmo
        st.session_state.x_col = []
        st.session_state.y_col = None

    # Selección de variables
    st.subheader("Selecciona las variables")
    columnas = df.columns.tolist()
    x_col = st.multiselect("Variables independientes (X)", columnas, default=st.session_state.x_col)
    y_col = st.selectbox("Variable dependiente (Y)", columnas, index=columnas.index(st.session_state.y_col) if st.session_state.y_col in columnas else 0)

    # Guardar en sesión
    st.session_state.x_col = x_col
    st.session_state.y_col = y_col

    if st.button("Ejecutar"):
        if len(x_col) == 0:
            st.error("Debes seleccionar al menos una variable independiente.")
        else:
            if algoritmo == "Random Forest":
                ejecutar_forest(df, x_col, y_col)
            elif algoritmo == "Red de Bayes":
                ejecutar_bayes(df, x_col, y_col)
