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
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import plotly.express as px
import plotly.graph_objects as go


# =======================
#   Utilidades de datos
# =======================

def _es_numerica(serie: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(serie)


def _prep_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas datetime a variables numéricas y elimina la original.
    - Detecta datetimes con 'datetime' y 'datetimetz'.
    - Intenta parsear columnas object si >60% son convertibles a datetime.
    - Quita tz y genera year, month, day, hour y epoch (segundos).
    """
    # 1) Detectar ya tipadas
    dt_cols = list(X.select_dtypes(include=["datetime"]).columns) + \
              list(X.select_dtypes(include=["datetimetz"]).columns)

    # 2) Intentar parsear object a datetime
    for c in X.select_dtypes(include=["object"]).columns:
        parsed = pd.to_datetime(X[c], errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() > 0.6:
            X[c] = parsed
            if c not in dt_cols:
                dt_cols.append(c)

    # 3) Expandir cada datetime
    for c in dt_cols:
        dt = X[c]

        # quitar tz si existe
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
    """
    Selecciona columnas, convierte datetime, aplica one-hot,
    fuerza a numérico y rellena NaN con la mediana.
    """
    data = df[x_col + [y_col]].dropna(subset=[y_col]).copy()

    X = data[x_col].copy()
    X = _prep_datetime_features(X)

    # One-hot para categóricas
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
    """Se considera regresión si Y es numérica y tiene muchas clases (>20)."""
    return _es_numerica(y) and y.nunique() > 20


def _asegurar_clasificacion(y: pd.Series) -> Tuple[pd.Series, str]:
    """
    Si Y es continua con muchas clases, la discretiza en quantiles para clasificación.
    Devuelve Y como string y una nota explicativa.
    """
    msg = ""
    if _es_numerica(y) and y.nunique() > 20:
        for q in (5, 4, 3):
            try:
                y_binned = pd.qcut(y, q=q, duplicates="drop")
                msg = f"Y continua → discretizada en {len(y_binned.unique())} quantiles para clasificación."
                return y_binned.astype(str), msg
            except Exception:
                continue
        # Fallback a bins uniformes
        y_binned = pd.cut(y, bins=3)
        msg = "Y continua → discretizada en 3 bins uniformes para clasificación."
        return y_binned.astype(str), msg
    return y.astype(str), msg


def _safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Usa estratificación solo si TODAS las clases tienen ≥2 muestras.
    Si Y tiene solo una clase, devuelve None.
    """
    if y.nunique() < 2:
        return None
    vc = y.value_counts()
    stratify_arg = y if (vc >= 2).all() else None
    if stratify_arg is None:
        st.warning("Estratificación desactivada: hay clases con 1 sola muestra.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)


# =======================
#   Visualizaciones
# =======================

def _plot_feature_importance(model, X: pd.DataFrame, top_n: int = 10):
    importancias = getattr(model, "feature_importances_", None)
    if importancias is None:
        return

    imp_df = pd.DataFrame({"feature": X.columns, "importance": importancias})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    fig = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Importancia de Características – Random Forest",
    )
    fig.update_layout(
        xaxis_title="Importancia relativa",
        yaxis_title="Características",
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Las barras más largas indican variables con mayor influencia en las predicciones del modelo."
    )


def _plot_regression_charts(model, X_test, y_test, y_pred):
    """
    Gráficas de regresión:
    - Actual vs Predicho con banda de error
    - Histograma de residuales
    - Residuales vs Predicho
    """
    df = pd.DataFrame({"y_real": y_test, "y_pred": y_pred})

    # ---------- bandas de error con desviación estándar entre árboles ----------
    all_tree_preds = np.array([est.predict(X_test) for est in model.estimators_])
    pred_std = all_tree_preds.std(axis=0)

    min_val = float(min(df["y_real"].min(), df["y_pred"].min()))
    max_val = float(max(df["y_real"].max(), df["y_pred"].max()))

    fig1 = go.Figure()

    # dispersión
    fig1.add_trace(
        go.Scatter(
            x=df["y_real"],
            y=df["y_pred"],
            mode="markers",
            name="Predicciones",
            marker=dict(size=7, opacity=0.7),
        )
    )

    # línea y = x
    fig1.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="y = x (predicción perfecta)",
            line=dict(color="green", dash="dash"),
        )
    )

    # banda superior
    fig1.add_trace(
        go.Scatter(
            x=df["y_real"],
            y=df["y_pred"] + pred_std,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )

    # banda inferior rellena
    fig1.add_trace(
        go.Scatter(
            x=df["y_real"],
            y=df["y_pred"] - pred_std,
            fill="tonexty",
            mode="lines",
            name="Banda de error (±1σ)",
            line=dict(width=0),
            opacity=0.2,
        )
    )

    fig1.update_layout(
        title="Actual vs Predicho – Random Forest (Regresión)",
        xaxis_title="Valor real",
        yaxis_title="Valor predicho",
        template="plotly_white",
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "**Dispersión entre valores reales y predichos.**  \n"
        "La banda sombreada representa la variabilidad del modelo (±1 desviación estándar entre árboles). "
        "Una mayor cercanía de los puntos a la línea diagonal indica mejor desempeño."
    )

    # ---------- Histograma de residuales ----------
    resid = y_test - y_pred
    fig2 = px.histogram(
        resid,
        nbins=20,
        title="Distribución de Errores (Residuales)",
    )
    fig2.update_layout(
        xaxis_title="Residual (y_real - y_pred)",
        yaxis_title="Frecuencia",
        template="plotly_white",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "**Distribución de los errores (residuales).**  \n"
        "Una forma aproximadamente simétrica y centrada en cero indica que el modelo no presenta sesgo sistemático."
    )

    # ---------- Residuales vs Predicho ----------
    df3 = pd.DataFrame({"residual": resid, "y_pred": y_pred})
    fig3 = px.scatter(
        df3,
        x="y_pred",
        y="residual",
        title="Residuales vs Valor Predicho",
    )
    fig3.add_hline(y=0, line_dash="dash")
    fig3.update_layout(
        xaxis_title="Valor predicho",
        yaxis_title="Residual",
        template="plotly_white",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "**Relación entre residuales y valores predichos.**  \n"
        "Un patrón aleatorio alrededor de la línea 0 sugiere que la variabilidad del error es razonablemente constante."
    )


def _plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray):
    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig = px.imshow(
        cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Predicción", y="Valor real", color="Frecuencia"),
        title="Matriz de Confusión – Random Forest (Clasificación)",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Matriz de confusión.**  \n"
        "La diagonal principal muestra las predicciones correctas. "
        "Valores altos fuera de la diagonal indican clases que el modelo confunde con otras."
    )


def _plot_roc_pr_curves(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Curvas ROC y Precision–Recall.
    - Binaria: una curva ROC y una curva PR.
    - Multiclase: curvas por clase (one-vs-rest).
    """
    if not hasattr(model, "predict_proba"):
        st.info("El modelo no expone predict_proba; se omiten curvas ROC/PR.")
        return

    classes = np.unique(y_test)
    proba = model.predict_proba(X_test)

    try:
        if len(classes) == 2:
            # ---------- Binaria ----------
            y_bin = label_binarize(y_test, classes=classes).ravel()
            proba_pos = proba[:, 1]

            # ROC
            fpr, tpr, _ = roc_curve(y_bin, proba_pos)
            roc_auc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC (AUC = {roc_auc:.3f})",
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Azar",
                    line=dict(dash="dash"),
                )
            )
            fig_roc.update_layout(
                title="Curva ROC – Clasificación Binaria",
                xaxis_title="Tasa de Falsos Positivos (FPR)",
                yaxis_title="Tasa de Verdaderos Positivos (TPR)",
                template="plotly_white",
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption(
                "**Curva ROC.**  \n"
                "Mide la capacidad del modelo para distinguir entre clases. "
                "Un AUC cercano a 1 indica excelente separación, mientras que 0.5 equivale al azar."
            )

            # Precision–Recall
            prec, rec, _ = precision_recall_curve(y_bin, proba_pos)
            ap = average_precision_score(y_bin, proba_pos)

            fig_pr = go.Figure()
            fig_pr.add_trace(
                go.Scatter(
                    x=rec,
                    y=prec,
                    mode="lines",
                    name=f"PR (AP = {ap:.3f})",
                )
            )
            fig_pr.update_layout(
                title="Curva Precision–Recall – Clasificación Binaria",
                xaxis_title="Recall",
                yaxis_title="Precision",
                template="plotly_white",
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            st.caption(
                "**Curva Precision–Recall.**  \n"
                "Especialmente útil con clases desbalanceadas. "
                "Valores altos de precisión y recall indican buen desempeño en la clase positiva."
            )

        else:
            # ---------- Multiclase ----------
            y_bin = label_binarize(y_test, classes=classes)

            # ROC por clase
            fig_roc = go.Figure()
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"Clase {cls} (AUC={roc_auc:.2f})",
                    )
                )

            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Azar",
                    line=dict(dash="dash"),
                )
            )
            fig_roc.update_layout(
                title="Curvas ROC por Clase – Multiclase",
                xaxis_title="FPR",
                yaxis_title="TPR",
                template="plotly_white",
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption(
                "**Curvas ROC por clase.**  \n"
                "Permiten evaluar qué tan bien se distingue cada clase frente al resto."
            )

            # Precision–Recall por clase
            fig_pr = go.Figure()
            for i, cls in enumerate(classes):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
                ap = average_precision_score(y_bin[:, i], proba[:, i])
                fig_pr.add_trace(
                    go.Scatter(
                        x=rec,
                        y=prec,
                        mode="lines",
                        name=f"Clase {cls} (AP={ap:.2f})",
                    )
                )
            fig_pr.update_layout(
                title="Curvas Precision–Recall por Clase – Multiclase",
                xaxis_title="Recall",
                yaxis_title="Precision",
                template="plotly_white",
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            st.caption(
                "**Curvas Precision–Recall por clase.**  \n"
                "Muestran el compromiso entre precisión y cobertura (recall) para cada clase."
            )

    except Exception as e:
        st.info(f"No fue posible generar curvas ROC/PR (muestras muy pocas o clases problemáticas). Detalle: {e}")


# =======================
#   Entrada principal
# =======================

def ejecutar(df: pd.DataFrame, x_col: list, y_col: str):
    st.subheader("Resultados - Random Forest")

    if len(x_col) == 0:
        st.error("Debes seleccionar al menos una variable independiente (X).")
        return
    if y_col is None:
        st.error("Debes seleccionar la variable dependiente (Y).")
        return

    # Opciones de visualización
    with st.expander("Opciones de visualización", expanded=False):
        top_n_importance = st.slider("Top-N importancia de características", 5, 30, 10, step=1)
        test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.4, 0.2, step=0.05)

    # Preparar datos
    X, y_raw = _preparar_xy(df, x_col, y_col)
    es_reg = _es_regresion(y_raw)

    # ====================
    #       REGRESIÓN
    # ====================
    if es_reg:
        y = pd.to_numeric(y_raw, errors="coerce")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown("### Tipo de tarea: Regresión")

        # Métricas
        st.markdown("### Métricas del modelo")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² (Score)", f"{r2_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
        with col3:
            mse = mean_squared_error(y_test, y_pred)
            st.metric("MSE", f"{mse:.4f}")
        with col4:
            rmse = np.sqrt(mse)
            st.metric("RMSE", f"{rmse:.4f}")

        # Gráficas
        st.markdown("### Gráficas")
        _plot_regression_charts(model, X_test, y_test, y_pred)

        # Importancia
        st.markdown("### Importancia de características")
        _plot_feature_importance(model, X, top_n=top_n_importance)

    # ====================
    #    CLASIFICACIÓN
    # ====================
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

        st.markdown("### Tipo de tarea: Clasificación")

        # Métrica principal
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")
        if nota:
            st.info(nota)

        # Gráficas
        st.markdown("### Gráficas")
        _plot_confusion_matrix(y_test, y_pred)
        _plot_roc_pr_curves(model, X_test, y_test)

        # Reporte de clasificación
        st.markdown("### Reporte de clasificación")
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)

        # Importancia
        st.markdown("### Importancia de características")
        _plot_feature_importance(model, X, top_n=top_n_importance)
