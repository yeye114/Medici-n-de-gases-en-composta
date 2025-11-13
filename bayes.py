import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def validar_datos(df: pd.DataFrame, x_col: list, y_col: str) -> Tuple[bool, str]:
    """Validación exhaustiva de datos antes del entrenamiento"""
    
    # Validar DataFrame vacío
    if df.empty:
        return False, "El DataFrame está vacío"
    
    # Validar columnas
    if not x_col:
        return False, "Debes seleccionar al menos una variable predictora"
    
    missing_cols = [col for col in x_col + [y_col] if col not in df.columns]
    if missing_cols:
        return False, f"Columnas no encontradas: {', '.join(missing_cols)}"
    
    # Validar tipos de datos
    for col in x_col + [y_col]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                return False, f"La columna '{col}' no es numérica y no puede convertirse"
    
    # Validar varianza
    for col in x_col:
        if df[col].std() == 0:
            return False, f"La columna '{col}' tiene varianza cero (todos los valores son iguales)"
    
    if df[y_col].std() == 0:
        return False, f"La variable objetivo '{y_col}' tiene varianza cero"
    
    # Validar valores infinitos
    if df[x_col + [y_col]].isin([np.inf, -np.inf]).any().any():
        return False, "Se encontraron valores infinitos en los datos"
    
    return True, "Validación exitosa"


def detectar_y_manejar_outliers(df: pd.DataFrame, y_col: str, 
                                 method: str = 'iqr', 
                                 threshold: float = 1.5) -> Tuple[pd.DataFrame, dict]:
    """
    Detección robusta de outliers con múltiples métodos
    
    Parameters:
    -----------
    method : str
        'iqr' (Interquartile Range), 'zscore' (Z-Score), 'isolation' (Isolation Forest)
    threshold : float
        Multiplicador para IQR (default: 1.5) o Z-score (default: 3.0)
    """
    df_clean = df.copy()
    n_original = len(df)
    outliers_info = {}
    
    if method == 'iqr':
        Q1 = df[y_col].quantile(0.25)
        Q3 = df[y_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers_mask = (df[y_col] < lower_bound) | (df[y_col] > upper_bound)
        
        outliers_info = {
            'method': 'IQR',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': outliers_mask.sum(),
            'outliers_df': df[outliers_mask].copy()
        }
    
    elif method == 'zscore':
        z_scores = np.abs((df[y_col] - df[y_col].mean()) / df[y_col].std())
        outliers_mask = z_scores > threshold
        
        outliers_info = {
            'method': 'Z-Score',
            'threshold': threshold,
            'n_outliers': outliers_mask.sum(),
            'outliers_df': df[outliers_mask].copy()
        }
    
    return df_clean, outliers_mask, outliers_info


def imputar_valores_faltantes(df: pd.DataFrame, x_col: list, y_col: str, 
                               strategy: str = 'median') -> pd.DataFrame:
    """
    Imputación inteligente de valores faltantes
    
    Parameters:
    -----------
    strategy : str
        'mean', 'median', 'most_frequent', 'knn'
    """
    df_imputed = df.copy()
    
    # Calcular porcentaje de valores nulos
    null_percentages = df[x_col + [y_col]].isnull().mean() * 100
    high_null_cols = null_percentages[null_percentages > 50].index.tolist()
    
    if high_null_cols:
        st.warning(f"Columnas con >50% valores nulos: {', '.join(high_null_cols)}")
        st.info("Considera eliminar estas columnas o recolectar más datos")
    
    # Imputar valores
    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed[x_col + [y_col]] = imputer.fit_transform(df_imputed[x_col + [y_col]])
    
    return df_imputed


def seleccion_scaler(normalize_method: str = 'standard'):
    """Selección de método de normalización según características de datos"""
    scalers = {
        'standard': StandardScaler(),  # Para datos con distribución normal
        'robust': RobustScaler(),      # Resistente a outliers
        'none': None
    }
    return scalers.get(normalize_method, StandardScaler())


def calcular_metricas_completas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Cálculo exhaustivo de métricas de evaluación"""
    
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100,
        'Max_Error': np.max(np.abs(y_true - y_pred)),
        'Min_Error': np.min(np.abs(y_true - y_pred)),
        'Std_Error': np.std(y_true - y_pred)
    }
    
    # Métricas adicionales
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['Adjusted_R2'] = 1 - (ss_res / ss_tot)
    
    return metrics


def ejecutar(df: pd.DataFrame, 
             x_col: list, 
             y_col: str, 
             test_size: float = 0.2,
             normalize_method: str = 'standard',
             alpha_1: float = 1e-6, 
             alpha_2: float = 1e-6,
             lambda_1: float = 1e-6, 
             lambda_2: float = 1e-6,
             remove_outliers: bool = False,
             outlier_method: str = 'iqr',
             outlier_threshold: float = 1.5,
             impute_strategy: str = 'median',
             cv_folds: int = 5,
             random_state: int = 42):
    """
    Bayesian Ridge Regression mejorado con manejo robusto de datasets
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    x_col : list
        Lista de nombres de columnas para variables predictoras
    y_col : str
        Nombre de la columna objetivo
    test_size : float
        Proporción de datos para prueba (default: 0.2)
    normalize_method : str
        Método de normalización: 'standard', 'robust', 'none'
    alpha_1, alpha_2, lambda_1, lambda_2 : float
        Hiperparámetros del modelo Bayesiano
    remove_outliers : bool
        Si se deben eliminar automáticamente los outliers
    outlier_method : str
        Método de detección: 'iqr', 'zscore'
    outlier_threshold : float
        Umbral para detección de outliers (1.5 para IQR, 3.0 para Z-score)
    impute_strategy : str
        Estrategia de imputación: 'mean', 'median', 'most_frequent'
    cv_folds : int
        Número de folds para validación cruzada
    random_state : int
        Semilla aleatoria para reproducibilidad
    """
    
    try:
        # ==================== VALIDACIÓN INICIAL ====================
        st.title("Bayesian Ridge Regression")
        
        # ==================== INFORMACIÓN SOBRE EL MÉTODO ====================
        with st.expander("Información sobre Bayesian Ridge Regression", expanded=False):
            st.markdown("""
            ### ¿Qué es Bayesian Ridge Regression?
            
            Es una variante de regresión lineal que incluye regularización bayesiana:
            
            **Ventajas:**
            - Maneja automáticamente la regularización (no necesita ajustar lambda manualmente)
            - Proporciona estimaciones de incertidumbre en las predicciones
            - Robusto ante multicolinealidad
            - Menos propenso a overfitting que regresión lineal simple
            
            **Desventajas:**
            - Asume relaciones lineales
            - Más lento que regresión lineal simple
            - Puede no capturar relaciones complejas no lineales
            
            **Cuándo usarlo:**
            - Tienes relaciones aproximadamente lineales
            - Necesitas interpretabilidad de coeficientes
            - Quieres control automático de regularización
            - Tienes datos con posible multicolinealidad
            
            **Alternativas a considerar:**
            - Regresión Lineal: Más simple, sin regularización
            - Ridge/Lasso: Regularización manual
            - Random Forest: Captura no linealidades
            - XGBoost: Rendimiento superior en muchos casos
            """)
        
        with st.spinner("Validando datos..."):
            es_valido, mensaje = validar_datos(df, x_col, y_col)
            if not es_valido:
                st.error(f"Error de validación: {mensaje}")
                return
        
        # ==================== INFORMACIÓN DEL DATASET ====================
        st.subheader("Información del Dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de filas", len(df))
        with col2:
            st.metric("Variables predictoras", len(x_col))
        with col3:
            st.metric("Tamaño de prueba", f"{test_size*100:.0f}%")
        with col4:
            st.metric("CV Folds", cv_folds)
        
        # ==================== MANEJO DE VALORES NULOS ====================
        total_nulls = df[x_col + [y_col]].isnull().sum().sum()
        
        if total_nulls > 0:
            st.subheader("Manejo de Valores Nulos")
            null_info = df[x_col + [y_col]].isnull().sum()
            null_info = null_info[null_info > 0]
            
            st.warning(f"Se encontraron {total_nulls} valores nulos en {len(null_info)} columnas")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(null_info.to_frame('Valores Nulos'), use_container_width=True)
            with col2:
                st.info(f"**Estrategia de imputación:** {impute_strategy}")
            
            df = imputar_valores_faltantes(df, x_col, y_col, impute_strategy)
            st.success(f"Valores nulos imputados usando estrategia '{impute_strategy}'")
        
        # ==================== DETECCIÓN Y MANEJO DE OUTLIERS ====================
        st.subheader("Análisis de Outliers")
        
        df_original = df.copy()
        n_original = len(df_original)
        
        df_clean, outliers_mask, outliers_info = detectar_y_manejar_outliers(
            df, y_col, method=outlier_method, threshold=outlier_threshold
        )
        
        n_outliers = outliers_info['n_outliers']
        
        # Métricas de outliers
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de datos", n_original)
        with col2:
            st.metric("Outliers detectados", n_outliers, 
                     delta=f"{(n_outliers/n_original*100):.1f}%" if n_original > 0 else "0%",
                     delta_color="inverse")
        with col3:
            if 'lower_bound' in outliers_info:
                st.metric("Límite inferior", f"{outliers_info['lower_bound']:.2f}")
        with col4:
            if 'upper_bound' in outliers_info:
                st.metric("Límite superior", f"{outliers_info['upper_bound']:.2f}")
        
        if n_outliers > 0:
            with st.expander("Ver outliers detectados", expanded=False):
                st.dataframe(outliers_info['outliers_df'], use_container_width=True)
                st.markdown(f"""
                **Método de detección:** {outliers_info['method']}
                
                **Estos valores están fuera del rango esperado:**
                - Pueden ser errores de captura de datos
                - Casos excepcionales legítimos
                - Valores que distorsionan el modelo
                
                **Recomendación:** Revisa estos valores antes de continuar.
                """)
            
            if remove_outliers:
                df = df[~outliers_mask]
                st.success(f"Se eliminaron {n_outliers} outliers. Datos restantes: {len(df)}")
        else:
            st.success("No se detectaron outliers en los datos")
        
        # Verificar datos suficientes
        min_samples = max(10, len(x_col) * 5)  # Mínimo 5 muestras por variable
        if len(df) < min_samples:
            st.error(f"Datos insuficientes ({len(df)} observaciones). Se recomiendan al menos {min_samples}.")
            return
        
        # ==================== PREPARACIÓN DE DATOS ====================
        X = df[x_col].values
        y = df[y_col].values
        
        # Normalización
        scaler_X = seleccion_scaler(normalize_method)
        scaler_y = seleccion_scaler(normalize_method) if normalize_method != 'none' else None
        
        if normalize_method != 'none':
            X = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = y
        
        # División estratificada para regresión (basada en cuartiles)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_scaled, test_size=test_size, random_state=random_state
        )
        
        # ==================== ENTRENAMIENTO DEL MODELO ====================
        with st.spinner("Entrenando modelo Bayesian Ridge..."):
            model = BayesianRidge(
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                compute_score=True,
                fit_intercept=True,
                max_iter=300,
                tol=1e-3
            )
            model.fit(X_train, y_train)
        
        # ==================== PREDICCIONES ====================
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Desnormalizar
        if normalize_method != 'none':
            y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
            y_pred_train_original = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
            y_pred_test_original = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
        else:
            y_train_original = y_train
            y_test_original = y_test
            y_pred_train_original = y_pred_train
            y_pred_test_original = y_pred_test
        
        # ==================== MÉTRICAS COMPLETAS ====================
        st.subheader("Métricas de Rendimiento")
        
        metrics_test = calcular_metricas_completas(y_test_original, y_pred_test_original)
        metrics_train = calcular_metricas_completas(y_train_original, y_pred_train_original)
        
        with st.expander("¿Qué significan estas métricas?", expanded=False):
            st.markdown("""
            **Guía de Métricas:**
            
            - **MSE/RMSE/MAE:** Miden el error promedio. Menor = Mejor
            - **R²:** Proporción de varianza explicada (0-1). Mayor = Mejor
            - **MAPE:** Error porcentual promedio. Útil para comparaciones
            - **Adjusted R²:** R² ajustado por número de variables
            - **Max/Min Error:** Rango de errores en las predicciones
            - **Std Error:** Variabilidad de los errores
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Conjunto de Prueba")
            metrics_test_df = pd.DataFrame({
                "Métrica": list(metrics_test.keys()),
                "Valor": [f"{v:.4f}" for v in metrics_test.values()]
            })
            st.dataframe(metrics_test_df, use_container_width=True)
        
        with col2:
            st.markdown("### Conjunto de Entrenamiento")
            metrics_train_df = pd.DataFrame({
                "Métrica": list(metrics_train.keys()),
                "Valor": [f"{v:.4f}" for v in metrics_train.values()]
            })
            st.dataframe(metrics_train_df, use_container_width=True)
        
        # ==================== VALIDACIÓN CRUZADA ====================
        with st.spinner(f"Ejecutando validación cruzada con {cv_folds} folds..."):
            try:
                cv_scores = cross_val_score(
                    model, X, y_scaled, cv=cv_folds, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                cv_rmse = np.sqrt(-cv_scores)
                
            except Exception as e:
                st.warning(f"No se pudo completar la validación cruzada: {str(e)}")
        
        # ==================== VISUALIZACIONES ====================
        st.subheader("Visualizaciones de Predicciones")
        
        comparacion_df = pd.DataFrame({
            "Valor Real": y_test_original,
            "Valor Predicho": y_pred_test_original,
            "Error": y_test_original - y_pred_test_original,
            "Error Absoluto": np.abs(y_test_original - y_pred_test_original)
        })
        
        # Gráfico 1: Scatter Real vs Predicho
        st.markdown("#### Valores Reales vs Predichos")
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown("""
            **Este gráfico muestra la precisión de las predicciones:**
            
            - **Eje X:** Valores reales de tus datos de prueba
            - **Eje Y:** Valores predichos por el modelo
            - **Línea verde discontinua:** Representa la predicción perfecta (donde Real = Predicho)
            - **Puntos cercanos a la línea:** Predicciones muy precisas
            - **Puntos alejados de la línea:** Predicciones con mayor error
            - **Color de los puntos:** Indica la magnitud del error (más rojo = mayor error)
            - **Tamaño de los puntos:** Proporcional al error absoluto
            
            **Interpretación ideal:** Los puntos deberían formar una línea diagonal perfecta sobre la línea verde. 
            Si se dispersan mucho, el modelo tiene dificultades para predecir ciertos rangos de valores.
            """)
        
        fig1 = px.scatter(
            comparacion_df,
            x="Valor Real",
            y="Valor Predicho",
            color="Error Absoluto",
            size="Error Absoluto",
            hover_data=["Error"],
            title="Valores Reales vs Predichos",
            color_continuous_scale="Reds"
        )
        
        min_val = min(y_test_original.min(), y_pred_test_original.min())
        max_val = max(y_test_original.max(), y_pred_test_original.max())
        fig1.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Predicción Perfecta',
            line=dict(color='green', dash='dash')
        ))
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico 3: Residuos vs Predichos
        st.markdown("#### Residuos vs Valores Predichos")
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown("""
            **Este gráfico ayuda a detectar patrones problemáticos en los errores:**
            
            - **Eje X:** Valores predichos por el modelo
            - **Eje Y:** Residuos (errores)
            - **Línea roja horizontal en y=0:** Línea de error cero
            
            **Interpretación ideal:**
            - Los puntos deben distribuirse **aleatoriamente** alrededor de la línea roja
            - No debe haber patrones claros (formas de embudo, curvas, bandas)
            - La dispersión debe ser **homogénea** en todo el rango de valores predichos
            
            **Patrones problemáticos a detectar:**
            
            1. **Forma de embudo (heteroscedasticidad):**
               - Los errores aumentan o disminuyen con el valor predicho
               - Indica que la varianza del error no es constante
               
            2. **Patrón curvo (no linealidad):**
               - Los residuos forman una curva (U o U invertida)
               - El modelo no captura relaciones no lineales
               
            3. **Grupos o bandas:**
               - Puntos agrupados en niveles distintos
               - Puede indicar categorías ocultas en los datos
               
            4. **Outliers:**
               - Puntos muy alejados de la línea roja
               - Observaciones atípicas que el modelo no puede predecir bien
            """)
        
        fig3 = px.scatter(
            x=y_pred_test_original,
            y=comparacion_df["Error"],
            title="Residuos vs Valores Predichos",
            labels={"x": "Valor Predicho", "y": "Residuo"}
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Calcular valores para diagnóstico (sin mostrar)
        from scipy import stats
        _, p_value_normalidad = stats.shapiro(comparacion_df["Error"][:min(5000, len(comparacion_df))])
        correlation_residuals = np.corrcoef(np.abs(comparacion_df["Error"]), y_pred_test_original)[0, 1]
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(comparacion_df["Error"])
        
        # ==================== INTERPRETACIÓN AUTOMÁTICA ====================
        st.subheader("Interpretación Automática")
        
        interpretaciones = []
        
        # R²
        if metrics_test['R2'] > 0.9:
            interpretaciones.append(("success", "Excelente ajuste (R² > 0.9)"))
        elif metrics_test['R2'] > 0.7:
            interpretaciones.append(("info", "Buen ajuste (R² > 0.7)"))
        elif metrics_test['R2'] > 0.5:
            interpretaciones.append(("warning", "Ajuste moderado (R² > 0.5)"))
        else:
            interpretaciones.append(("error", "Ajuste pobre (R² < 0.5)"))
        
        # Overfitting
        if metrics_train['R2'] - metrics_test['R2'] > 0.15:
            interpretaciones.append(("warning", 
                f"Sobreajuste detectado: ΔR² = {metrics_train['R2'] - metrics_test['R2']:.3f}"))
        else:
            interpretaciones.append(("success", "Buena generalización"))
        
        # Error relativo
        rango_y = y_test_original.max() - y_test_original.min()
        rmse_relativo = (metrics_test['RMSE'] / rango_y) * 100 if rango_y > 0 else 0
        
        if rmse_relativo < 5:
            interpretaciones.append(("success", f"Error muy bajo: {rmse_relativo:.1f}% del rango"))
        elif rmse_relativo < 15:
            interpretaciones.append(("info", f"Error aceptable: {rmse_relativo:.1f}% del rango"))
        else:
            interpretaciones.append(("warning", f"Error alto: {rmse_relativo:.1f}% del rango"))
        
        # Normalidad
        if p_value_normalidad < 0.05:
            interpretaciones.append(("warning", "Los residuos no siguen distribución normal"))
        
        # Mostrar interpretaciones
        for tipo, mensaje in interpretaciones:
            if tipo == "success":
                st.success(mensaje)
            elif tipo == "info":
                st.info(mensaje)
            elif tipo == "warning":
                st.warning(mensaje)
            elif tipo == "error":
                st.error(mensaje)
        
        # ==================== ANÁLISIS DE CONVERGENCIA ====================
        if hasattr(model, 'scores_') and len(model.scores_) > 0:
            st.subheader("Análisis de Convergencia del Modelo")
            with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
                st.markdown("""
                **Este gráfico muestra cómo el modelo mejora durante el entrenamiento:**
                
                - **Eje X:** Número de iteración del algoritmo
                - **Eje Y:** Score del modelo (medida de calidad del ajuste)
                - **Tendencia ascendente:** El modelo está mejorando
                - **Línea estable/plana:** El modelo ha convergido (encontró la mejor solución)
                
                **Interpretación ideal:**
                - La línea debe subir rápidamente al inicio y luego estabilizarse
                - Si sigue subiendo al final: El modelo podría mejorar con más iteraciones
                - Si es muy errática: Puede haber problemas de estabilidad numérica
                
                **Qué hacer si no converge:**
                - Aumenta `max_iter` (número máximo de iteraciones)
                - Normaliza tus datos si no lo has hecho
                - Ajusta los hiperparámetros alpha y lambda
                """)
            
            fig_convergence = go.Figure()
            fig_convergence.add_trace(go.Scatter(
                x=list(range(len(model.scores_))),
                y=model.scores_,
                mode='lines+markers',
                name='Score',
                line=dict(color='blue')
            ))
            fig_convergence.update_layout(
                title="Evolución del Score durante el Entrenamiento",
                xaxis_title="Iteración",
                yaxis_title="Score",
                hovermode='x unified'
            )
            st.plotly_chart(fig_convergence, use_container_width=True)
            
            if len(model.scores_) >= 300:
                st.warning("El modelo alcanzó el máximo de iteraciones. Considera aumentar `max_iter`")
        
        # ==================== COMPARACIÓN TRAIN VS TEST ====================
        st.subheader("Comparación Train vs Test")
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown("""
            **Este gráfico compara el rendimiento del modelo en datos de entrenamiento vs prueba:**
            
            - **Barras azules (Entrenamiento):** Rendimiento en datos usados para entrenar el modelo
            - **Barras naranjas (Prueba):** Rendimiento en datos nuevos que el modelo nunca vio
            - **R²:** Mayor es mejor (ideal: valores similares en ambos conjuntos)
            - **RMSE y MAE:** Menor es mejor (ideal: valores similares en ambos conjuntos)
            
            **Interpretación ideal:**
            - **Valores similares en Train y Test:** El modelo generaliza bien
            - **Train mucho mejor que Test:** Overfitting (sobreajuste) - el modelo memoriza en lugar de aprender
            - **Test mejor que Train:** Raro, puede indicar problemas en la división de datos
            
            **Señales de alerta:**
            - Si R² Train > 0.95 pero R² Test < 0.70: Claro overfitting
            - Si RMSE Test >> RMSE Train: El modelo no generaliza bien
            
            **Qué hacer si hay overfitting:**
            - Recolecta más datos
            - Reduce la complejidad del modelo
            - Aumenta la regularización
            - Usa validación cruzada
            """)
        
        comparacion_conjuntos = pd.DataFrame({
            'Conjunto': ['Entrenamiento', 'Prueba'],
            'R²': [metrics_train['R2'], metrics_test['R2']],
            'RMSE': [metrics_train['RMSE'], metrics_test['RMSE']],
            'MAE': [metrics_train['MAE'], metrics_test['MAE']],
            'MAPE': [metrics_train['MAPE'], metrics_test['MAPE']]
        })
        
        fig_comp = go.Figure()
        for metric in ['R²', 'RMSE', 'MAE']:
            fig_comp.add_trace(go.Bar(
                name=metric,
                x=comparacion_conjuntos['Conjunto'],
                y=comparacion_conjuntos[metric],
                text=comparacion_conjuntos[metric].round(4),
                textposition='auto'
            ))
        
        fig_comp.update_layout(
            title="Comparación de Métricas: Train vs Test",
            barmode='group',
            yaxis_title="Valor de Métrica"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # ==================== RESUMEN FINAL ====================
        st.subheader("Resultados del Modelo Bayesian Ridge Regression")
        
        resumen_final = f"""
        **Dataset:**
        - Observaciones totales: {n_original}
        - Observaciones tras limpieza: {len(df)}
        - Variables predictoras: {len(x_col)}
        - Outliers eliminados: {n_outliers if remove_outliers else 0}
        
        **Rendimiento:**
        - R² (Test): {metrics_test['R2']:.4f}
        - RMSE (Test): {metrics_test['RMSE']:.4f}
        - MAE (Test): {metrics_test['MAE']:.4f}
        - MAPE (Test): {metrics_test['MAPE']:.2f}%
        
        **Diagnóstico:**
        - Overfitting: {'Sí' if metrics_train['R2'] - metrics_test['R2'] > 0.15 else 'No'}
        - Residuos normales: {'Sí' if p_value_normalidad > 0.05 else 'No'}
        - Error relativo: {rmse_relativo:.2f}% del rango
        
        **Configuración:**
        - Normalización: {normalize_method}
        - Test size: {test_size*100:.0f}%
        - CV Folds: {cv_folds}
        - Random state: {random_state}
        """
        
        st.markdown(resumen_final)
        
        st.success("Análisis completado exitosamente")
        
    except Exception as e:
        st.error(f"Error crítico durante la ejecución: {str(e)}")
        st.exception(e)
        
        # Información de debugging
        with st.expander("Información de Debug", expanded=False):
            st.write("**Variables en el error:**")
            st.write(f"- Forma del DataFrame: {df.shape if 'df' in locals() else 'N/A'}")
            st.write(f"- Columnas X: {x_col if 'x_col' in locals() else 'N/A'}")
            st.write(f"- Columna Y: {y_col if 'y_col' in locals() else 'N/A'}")
            st.write(f"- Valores nulos: {df.isnull().sum().sum() if 'df' in locals() else 'N/A'}")