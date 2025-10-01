# visualizar.py
import io
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============ Utilidades de lectura ============

def _detectar_separador(sample_bytes: bytes) -> str:
    """Intenta detectar el separador usando csv.Sniffer; regresa coma si falla."""
    try:
        texto = sample_bytes.decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(texto, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","

def _leer_csv(
    file_bytes: bytes,
    encoding: str,
    separador: str,
    tiene_encabezado: bool,
    decimal: str,
) -> pd.DataFrame:
    # Autodetectar separador si corresponde
    if separador == "auto":
        separador = _detectar_separador(file_bytes[:4096])
    if separador == "\\t":  # convertir literal a TAB real
        separador = "\t"

    header = 0 if tiene_encabezado else None
    buffer = io.BytesIO(file_bytes)

    df = pd.read_csv(
        buffer,
        encoding=encoding,
        sep=separador,
        header=header,
        decimal=decimal,
        engine="python",
    )

    if header is None:
        df.columns = [f"col_{i}" for i in range(1, len(df.columns) + 1)]

    return df

# ============ Limpieza ============

def _es_candidata_fecha(nombre: str) -> bool:
    """Heurística simple para detectar columna de tiempo por nombre."""
    nombre = nombre.lower()
    hints = ["time", "fecha", "date", "timestamp", "datetime", "hora"]
    return any(h in nombre for h in hints)

def _convertir_a_datetime(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convierte columnas que parecen fechas a datetime (best effort)."""
    advertencias = []
    for col in df.columns:
        if _es_candidata_fecha(col) or df[col].dtype == "object":
            # Intento de parseo
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, utc=False)
            # Si hubo conversión exitosa (no todos NaT), la aplicamos
            if parsed.notna().mean() > 0.6:  # al menos 60% convertibles
                df[col] = parsed
            elif _es_candidata_fecha(col):
                advertencias.append(f"‘{col}’ no se pudo convertir completamente a fecha/hora.")
    return df, advertencias

def _limpieza_basica(df: pd.DataFrame, decimales: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Limpieza estándar:
    - strip a nombres de columnas
    - trim de strings
    - conversión a datetime para columnas candidatas
    - conversión a numérico cuando posible
    - eliminación de duplicados
    - redondeo de numéricos
    """
    notas = []

    # Normalizar nombres de columnas
    df.columns = [c.strip() for c in df.columns]

    # Trim de celdas string
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Convertir a datetime cuando tenga sentido
    df, adv = _convertir_a_datetime(df)
    notas.extend(adv)

    # Intento de convertir columnas object a numérico si parecen números
    for c in df.columns:
        if df[c].dtype == "object":
            # Reemplazo suave de coma decimal si aplica (sin romper textos)
            muestra = df[c].dropna().astype(str).head(50)
            # Si la mayoría cumple patrón numérico con coma o punto, probamos
            looks_numeric = (muestra.str.replace(",", ".", regex=False)
                                      .str.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", case=False).mean())
            if looks_numeric > 0.6:
                df[c] = pd.to_numeric(muestra.str.replace(",", ".", regex=False), errors="coerce")
                # Reaplicar a toda la columna
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Eliminar duplicados exactos
    dups = int(df.duplicated().sum())
    if dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        notas.append(f"Se eliminaron {dups} filas duplicadas.")

    # Redondeo de numéricos
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0 and decimales is not None:
        df[num_cols] = df[num_cols].round(decimales)

    return df, notas

# ============ UI principal ============

def _numericas(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["number"]).columns.tolist()

def mostrar():
    st.title("Visualizar información")
    st.write(
        "Sube tu CSV. Se limpiará automáticamente y los datos **limpios** se guardarán en "
        "`st.session_state.df` para que la sección **Análisis** los use directamente."
    )

    # Opciones de lectura
    with st.expander("Opciones de lectura del CSV", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        separador = c1.selectbox("Separador", ["auto", ",", ";", "\\t", "|"], index=0)
        encoding = c2.selectbox("Encoding", ["utf-8", "latin-1", "utf-16", "cp1252"], index=0)
        tiene_encabezado = c3.checkbox("Tiene encabezado", value=True)
        decimal = c4.selectbox("Separador de decimales", [".", ","], index=0)

    archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if archivo is None:
        st.info("Sube un archivo para comenzar.")
        return

    try:
        # Lectura
        file_bytes = archivo.read()
        df_raw = _leer_csv(
            file_bytes=file_bytes,
            encoding=encoding,
            separador=separador,
            tiene_encabezado=tiene_encabezado,
            decimal=decimal,
        )

        # Controles de limpieza
        st.subheader("Limpieza automática")
        cA, cB = st.columns([1, 1])
        decimales = cA.slider("Redondeo de columnas numéricas (decimales)", 0, 6, 2, step=1)
        aplicar = cB.checkbox("Aplicar limpieza automáticamente", value=True)

        if aplicar:
            df_clean, notas = _limpieza_basica(df_raw.copy(), decimales=decimales)
        else:
            df_clean, notas = df_raw.copy(), ["(Limpieza automática desactivada)"]

        # Guardar en sesión: DataFrame limpio
        st.session_state.df = df_clean
        st.session_state.filename = archivo.name

        # Mensajes
        st.success(f"Archivo cargado: **{archivo.name}**")
        st.caption(f"Dimensiones (limpio): {df_clean.shape[0]:,} filas × {df_clean.shape[1]:,} columnas")

        # Vista previa
        st.subheader("Vista previa del DataFrame limpio")
        nfilas = st.slider("Filas a mostrar", 5, 200, 20, step=5)
        st.dataframe(df_clean.head(nfilas), use_container_width=True)

        # Resumen
        with st.expander("Resumen y calidad de datos"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Tipos de datos**")
                tipos = pd.DataFrame({"columna": df_clean.columns, "dtype": df_clean.dtypes.astype(str).values})
                st.dataframe(tipos, use_container_width=True, hide_index=True)
            with c2:
                st.write("**Valores faltantes por columna**")
                miss = df_clean.isna().sum().to_frame(name="faltantes")
                st.dataframe(miss, use_container_width=True)

            st.write("**Estadísticos (numéricos)**")
            if len(_numericas(df_clean)) > 0:
                st.dataframe(df_clean.describe(include="number").T, use_container_width=True)
            else:
                st.info("No hay columnas numéricas para describir.")

            if notas:
                st.write("**Notas de limpieza**")
                for n in notas:
                    st.markdown(f"- {n}")

        # Gráficas rápidas
        st.subheader("Gráficas rápidas")
        graf = st.selectbox("Tipo de gráfica", ["Línea", "Barras", "Dispersión", "Histograma"])

        if graf in {"Línea", "Barras"}:
            usar_indice = st.checkbox("Usar índice como eje X", value=False)
            cols = df_clean.columns.tolist()
            x_col = None if usar_indice else st.selectbox("Eje X", cols, index=0)
            y_cols = st.multiselect("Series (Y)", _numericas(df_clean), default=_numericas(df_clean)[:1])

            if len(y_cols) == 0:
                st.info("Selecciona al menos una columna numérica para Y.")
            else:
                data_plot = df_clean.copy()
                if x_col is not None:
                    # Si la X es fecha/tiempo, conviene indexarla para mejor render
                    data_plot = data_plot.set_index(x_col)
                if graf == "Línea":
                    st.line_chart(data_plot[y_cols], use_container_width=True)
                else:
                    st.bar_chart(data_plot[y_cols], use_container_width=True)

        elif graf == "Dispersión":
            nums = _numericas(df_clean)
            if len(nums) < 2:
                st.info("Se requieren al menos dos columnas numéricas.")
            else:
                c1, c2 = st.columns(2)
                x = c1.selectbox("X", nums, index=0)
                y = c2.selectbox("Y", nums, index=1 if len(nums) > 1 else 0)
                st.scatter_chart(df_clean[[x, y]], x=x, y=y, use_container_width=True)

        else:  # Histograma
            nums = _numericas(df_clean)
            if len(nums) == 0:
                st.info("No hay columnas numéricas para histograma.")
            else:
                colh1, colh2 = st.columns([2, 1])
                col_hist = colh1.selectbox("Columna", nums, index=0)
                bins = colh2.slider("Bins", 5, 200, 30, step=5)
                serie = df_clean[col_hist].dropna().astype(float)
                conteo, bordes = np.histogram(serie, bins=bins)
                centros = (bordes[:-1] + bordes[1:]) / 2
                hist_df = pd.DataFrame({"bin_center": centros, "count": conteo})
                st.bar_chart(hist_df, x="bin_center", y="count", use_container_width=True)

        st.info("✅ Datos limpios disponibles. Ve a la pestaña **Análisis** para ejecutar los algoritmos.")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
        st.stop()
