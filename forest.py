# forest.py
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# -------------------- utilidades --------------------

def _es_numerica(serie: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(serie)


def _prep_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas datetime a variables numéricas y elimina la original.
    - Detecta datetimes con 'datetime' y 'datetimetz'.
    - Intenta parsear columnas object si >60% son convertibles a datetime.
    - Quita tz y genera year, month, day, hour y epoch (segundos).
    """
    # 1) Detectar datetimes ya tipadas
    dt_cols = list(X.select_dtypes(include=["datetime"]).columns) + \
              list(X.select_dtypes(include=["datetimetz"]).columns)

    # 2) Intentar parsear objects que parezcan fechas
    for c in X.select_dtypes(include=["object"]).columns:
        parsed = pd.to_datetime(X[c], errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() > 0.6:
            X[c] = parsed
            if c not in dt_cols:
                dt_cols.append(c)

    # 3) Expandir cada datetime a features numéricos
    for c in dt_cols:
        dt = X[c]

        # Quitar tz si existe
        try:
            if getattr(dt.dt, "tz", None) is not None:
                try:
                    dt = dt.dt.tz_convert(None)
                except Exception:
                    dt = dt.dt.tz_localize(None)
        except Exception:
            pass

        X[f"{c}_year"] = dt.dt.year
        X[f"{c}_month"] = dt.dt.month
        X[f"{c}_day"] = dt.dt.day
        X[f"{c}_hour"] = dt.dt.hour

        # Epoch (segundos). Fallback por compatibilidad.
        try:
            epoch = dt.view("int64") // 10**9
        except Exception:
            epoch = pd.to_datetime(dt, errors="coerce").view("int64") // 10**9
        X[f"{c}_epoch"] = epoch

        X.drop(columns=[c], inplace=True, errors="ignore")

    return X


def _preparar_xy(df: pd.DataFrame, x_col: list, y_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = df[x_col + [y_col]].dropna(subset=[y_col]).copy()

    X = data[x_col].copy()
    X = _prep_datetime_features(X)

    # One-hot para object/category
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Forzar todo a numérico
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Relleno de NaN
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    y = data[y_col]
    return X, y


def _es_regresion(y: pd.Series) -> bool:
    if not _es_numerica(y):
        return False
    return y.nunique() > 20


def _asegurar_clasificacion(y: pd.Series) -> Tuple[pd.Series, str]:
    msg = ""
    if _es_numerica(y) and y.nunique() > 20:
        for q in (5, 4, 3):
            try:
                y_binned = pd.qcut(y, q=q, duplicates="drop")
                msg = f"Y continua → discretizada en {len(y_binned.unique())} quantiles para clasificación."
                return y_binned.astype(str), msg
            except Exception:
                continue
        y_binned = pd.cut(y, bins=3)
        msg = "Y continua → discretizada en 3 bins uniformes para clasificación."
        return y_binned.astype(str), msg
    else:
        return y.astype(str), msg


def _safe_train_test_split(X, y, test_size=0.2, random_state=42):
    if y.nunique() < 2:
        return None
    vc = y.value_counts()
    stratify_arg = y if (vc >= 2).all() else None
    if stratify_arg is None:
        st.warning("Estratificación desactivada: hay clases con 1 sola muestra.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)


# -------------------- entrada principal --------------------

def ejecutar(df: pd.DataFrame, x_col: list, y_col: str):
    st.subheader("Resultados - Random Forest")

    if len(x_col) == 0:
        st.error("Debes seleccionar al menos una variable independiente (X).")
        return
    if y_col is None:
        st.error("Debes seleccionar la variable dependiente (Y).")
        return

    X, y_raw = _preparar_xy(df, x_col, y_col)
    es_regresion = _es_regresion(y_raw)

    # -------- Regresión --------
    if es_regresion:
        y = pd.to_numeric(y_raw, errors="coerce")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown("**Tarea detectada:** Regresión")
        st.write(f"R²: **{r2_score(y_test, y_pred):.4f}**")
        st.write(f"MAE: **{mean_absolute_error(y_test, y_pred):.4f}**")
        # --- Cambio aquí: RMSE sin parámetro 'squared' ---
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.write(f"RMSE: **{rmse:.4f}**")

    # -------- Clasificación --------
    else:
        y, nota = _asegurar_clasificacion(y_raw)
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        split = _safe_train_test_split(X, y, test_size=0.2, random_state=42)
        if split is None:
            st.error("Clasificación no posible: la variable objetivo (Y) solo tiene una clase.")
            return

        X_train, X_test, y_train, y_test = split
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown("**Tarea detectada:** Clasificación")
        if nota:
            st.info(nota)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: **{acc:.4f}**")

        st.write("**Reporte de clasificación**")
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)

        st.write("**Matriz de confusión**")
        labels_sorted = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        cm_df = pd.DataFrame(
            cm,
            index=[f"real:{c}" for c in labels_sorted],
            columns=[f"pred:{c}" for c in labels_sorted],
        )
        st.dataframe(cm_df, use_container_width=True)

    # -------- Importancia de características --------
    try:
        importancias = getattr(model, "feature_importances_", None)
        if importancias is not None:
            st.write("**Importancia de características (top 10)**")
            imp_df = pd.DataFrame({"feature": X.columns, "importance": importancias})
            imp_df = imp_df.sort_values("importance", ascending=False).head(10).reset_index(drop=True)
            st.dataframe(imp_df, use_container_width=True)
            st.bar_chart(imp_df.set_index("feature"))
    except Exception:
        pass
