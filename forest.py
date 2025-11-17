# forest.py
from typing import Tuple, List

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
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go


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
    dt_cols = list(X.select_dtypes(include=["datetime"]).columns) + \
              list(X.select_dtypes(include=["datetimetz"]).columns)

    for c in X.select_dtypes(include=["object"]).columns:
        parsed = pd.to_datetime(X[c], errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() > 0.6:
            X[c] = parsed
            if c not in dt_cols:
                dt_cols.append(c)

    for c in dt_cols:
        dt = X[c]
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
        try:
            epoch = dt.view("int64") // 10**9
        except Exception:
            epoch = pd.to_datetime(dt, errors="coerce").view("int64") // 10**9
        X[f"{c}_epoch"] = epoch

        X.drop(columns=[c], inplace=True, errors="ignore")

    return X


def _preparar_xy(df: pd.DataFrame, x_col: list, y_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Selecciona columnas, convierte datetime, one-hot, fuerza numérico y fillna."""
    data = df[x_col + [y_col]].dropna(subset=[y_col]).copy()

    X = data[x_col].copy()
    X = _prep_datetime_features(X)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    y = data[y_col]
    return X, y


def _es_regresion(y: pd.Series) -> bool:
    return _es_numerica(y) and y.nunique() > 20


def _asegurar_clasificacion(y: pd.Series) -> Tuple[pd.Series, str]:
    """Si y es continua con muchas clases, discretiza en quantiles."""
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
    return y.astype(str), msg


def _safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """Estratifica solo si TODAS las clases tienen ≥2 muestras; si hay 1 clase, devuelve None."""
    if y.nunique() < 2:
        return None
    vc = y.value_counts()
    stratify_arg = y if (vc >= 2).all() else None
    if stratify_arg is None:
        st.warning("Estratificación desactivada: hay clases con 1 sola muestra.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)


# -------------------- ayudas de visualización --------------------

def _plot_feature_importance(model, X: pd.DataFrame, top_n: int = 10):
    importancias = getattr(model, "feature_importances_", None)
    if importancias is None:
        return
    imp_df = pd.DataFrame({"feature": X.columns, "importance": importancias})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Importancia de características: mayor barra ⇒ mayor contribución al modelo.")

def _plot_regression_charts(y_test: pd.Series, y_pred: np.ndarray):
    # 1) Actual vs Predicho
    df1 = pd.DataFrame({"y_real": y_test, "y_pred": y_pred})
    # Sin trendline="ols" para no depender de statsmodels
    fig1 = px.scatter(df1, x="y_real", y="y_pred")

    # línea y = x para referencia perfecta
    min_val = min(df1["y_real"].min(), df1["y_pred"].min())
    max_val = max(df1["y_real"].max(), df1["y_pred"].max())
    fig1.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="y = x (predicción perfecta)",
            line=dict(dash="dash")
        )
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Dispersión Actual vs Predicho: los puntos cercanos a la línea y=x indican buenas predicciones."
    )

    # 2) Histograma de residuales
    resid = y_test - y_pred
    fig2 = px.histogram(resid, nbins=20)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Histograma de residuales: idealmente centrado en 0 y relativamente simétrico (sin sesgos fuertes)."
    )

    # 3) Residual vs Predicho
    df3 = pd.DataFrame({"residual": resid, "y_pred": y_pred})
    fig3 = px.scatter(df3, x="y_pred", y="residual")
    fig3.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Residuales vs Predicho: un patrón aleatorio alrededor de 0 sugiere que los errores son razonables."
    )

def _plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray):
    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig = px.imshow(cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                    color_continuous_scale="Blues", text_auto=True,
                    labels=dict(x="Predicción", y="Real", color="Frecuencia"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Matriz de confusión: diagonal alta indica buen desempeño (clases bien clasificadas).")

def _plot_roc_pr_curves(model, X_test: pd.DataFrame, y_test: pd.Series):
    """ROC y PR. Binaria: curva directa. Multiclase: macro/micro promedios."""
    # Probabilidades (si el modelo las soporta)
    if not hasattr(model, "predict_proba"):
        st.info("El modelo no expone predict_proba; se omiten curvas ROC/PR.")
        return

    classes = pd.unique(y_test)
    if len(classes) == 2:
        # Binaria
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test.astype(str), (proba > 0.5).astype(int), pos_label="1")
        # si las clases no son '0'/'1', volvemos a calcular usando y_test label-encoded
        try:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_test, classes=classes)[:, 0]
            fpr, tpr, _ = roc_curve(y_bin, proba)
        except Exception:
            pass
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar", line=dict(dash="dash")))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Curva ROC: cuanto más arriba/izquierda, mejor (AUC alto).")

        prec, rec, _ = precision_recall_curve(y_bin, proba)
        ap = average_precision_score(y_bin, proba)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP={ap:.3f}"))
        fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Curva Precision–Recall: útil con clases desbalanceadas; AP más alto es mejor.")
    else:
        # Multiclase: micro/macro
        from sklearn.preprocessing import label_binarize
        classes_sorted = sorted(classes)
        y_bin = label_binarize(y_test, classes=classes_sorted)
        proba = model.predict_proba(X_test)

        # ROC macro
        roc_aucs = []
        fig = go.Figure()
        for i, c in enumerate(classes_sorted):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_aucs.append(auc(fpr, tpr))
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Clase {c} (AUC {roc_aucs[-1]:.2f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar", line=dict(dash="dash")))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", title="ROC por clase")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Curvas ROC por clase (multiclase).")

        # PR micro
        fig2 = go.Figure()
        for i, c in enumerate(classes_sorted):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
            ap = average_precision_score(y_bin[:, i], proba[:, i])
            fig2.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"Clase {c} (AP {ap:.2f})"))
        fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Precision–Recall por clase")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Curvas Precision–Recall por clase (multiclase).")


# -------------------- entrada principal --------------------

def ejecutar(df: pd.DataFrame, x_col: list, y_col: str):
    st.subheader("Resultados - Random Forest")

    if len(x_col) == 0:
        st.error("Debes seleccionar al menos una variable independiente (X).")
        return
    if y_col is None:
        st.error("Debes seleccionar la variable dependiente (Y).")
        return

    # Config de visualización
    with st.expander("Opciones de visualización", expanded=False):
        top_n_importance = st.slider("Top-N importancia de características", 5, 30, 10, step=1)
        test_size = st.slider("Tamaño de test", 0.1, 0.4, 0.2, step=0.05)

    # Preparación de datos
    X, y_raw = _preparar_xy(df, x_col, y_col)
    es_regresion = _es_regresion(y_raw)

    # -------- Regresión --------
    if es_regresion:
        y = pd.to_numeric(y_raw, errors="coerce")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown("**Tarea detectada:** Regresión")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
        with col2: st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        with col3: st.metric("RMSE", f"{rmse:.4f}")

        st.markdown("### Gráficas")
        _plot_regression_charts(y_test, y_pred)

        st.markdown("### Importancia de características")
        _plot_feature_importance(model, X, top_n=top_n_importance)

    # -------- Clasificación --------
    else:
        y, nota = _asegurar_clasificacion(y_raw)
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        split = _safe_train_test_split(X, y, test_size=test_size, random_state=42)
        if split is None:
            st.error("Clasificación no posible: la variable objetivo (Y) solo tiene una clase.")
            return

        X_train, X_test, y_train, y_test = split
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown("**Tarea detectada:** Clasificación")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")
        if nota:
            st.info(nota)

        st.markdown("### Gráficas")
        _plot_confusion_matrix(y_test, y_pred)
        _plot_roc_pr_curves(model, X_test, y_test)

        st.markdown("### Reporte de clasificación")
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)

        st.markdown("### Importancia de características")
        _plot_feature_importance(model, X, top_n=top_n_importance)
