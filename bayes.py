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
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# ==================== FUNCIONES AUXILIARES ====================

def validar_datos(df: pd.DataFrame, x_col: list, y_col: str) -> Tuple[bool, str]:
    """Validación exhaustiva de datos antes del entrenamiento"""
    if df.empty:
        return False, "El DataFrame está vacío"
    
    if not x_col:
        return False, "Debes seleccionar al menos una variable predictora"
    
    missing_cols = [col for col in x_col + [y_col] if col not in df.columns]
    if missing_cols:
        return False, f"Columnas no encontradas: {', '.join(missing_cols)}"
    
    # Validar y convertir tipos de datos
    for col in x_col + [y_col]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                return False, f"La columna '{col}' no es numérica y no puede convertirse"
    
    # Validar varianza
    for col in x_col:
        if df[col].std() == 0:
            return False, f"La columna '{col}' tiene varianza cero"
    
    if df[y_col].std() == 0:
        return False, f"La variable objetivo '{y_col}' tiene varianza cero"
    
    # Validar valores infinitos
    if df[x_col + [y_col]].isin([np.inf, -np.inf]).any().any():
        return False, "Se encontraron valores infinitos en los datos"
    
    return True, "Validación exitosa"


def detectar_outliers(df: pd.DataFrame, y_col: str, method: str = 'iqr', 
                      threshold: float = 1.5) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """Detección de outliers con múltiples métodos"""
    outliers_info = {}
    
    if method == 'iqr':
        Q1, Q3 = df[y_col].quantile([0.25, 0.75])
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
    
    return df.copy(), outliers_mask, outliers_info


def imputar_valores_faltantes(df: pd.DataFrame, x_col: list, y_col: str, 
                               strategy: str = 'median') -> pd.DataFrame:
    """Imputación de valores faltantes"""
    df_imputed = df.copy()
    null_percentages = df[x_col + [y_col]].isnull().mean() * 100
    high_null_cols = null_percentages[null_percentages > 50].index.tolist()
    
    if high_null_cols:
        st.warning(f"Columnas con >50% valores nulos: {', '.join(high_null_cols)}")
        st.info("Considera eliminar estas columnas o recolectar más datos")
    
    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed[x_col + [y_col]] = imputer.fit_transform(df_imputed[x_col + [y_col]])
    
    return df_imputed


def seleccion_scaler(normalize_method: str = 'standard'):
    """Selección de método de normalización"""
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'none': None
    }
    return scalers.get(normalize_method, StandardScaler())


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Cálculo de métricas de evaluación"""
    mse = mean_squared_error(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100,
        'Max_Error': np.max(np.abs(y_true - y_pred)),
        'Min_Error': np.min(np.abs(y_true - y_pred)),
        'Std_Error': np.std(y_true - y_pred),
        'Adjusted_R2': 1 - (ss_res / ss_tot)
    }


# ==================== FUNCIÓN PRINCIPAL ====================

def ejecutar(df: pd.DataFrame, x_col: list, y_col: str, test_size: float = 0.7,
             normalize_method: str = 'standard', alpha_1: float = 1e-6, 
             alpha_2: float = 1e-6, lambda_1: float = 1e-6, lambda_2: float = 1e-6,
             remove_outliers: bool = False, outlier_method: str = 'iqr',
             outlier_threshold: float = 1.5, impute_strategy: str = 'median',
             cv_folds: int = 5, random_state: int = 42):
    """Bayesian Ridge Regression con análisis completo"""
    
    try:
        st.title("Bayesian Ridge Regression")
        
        # Información sobre el método
        with st.expander("Información sobre Bayesian Ridge Regression", expanded=False):
            st.markdown("""
            ### ¿Qué es Bayesian Ridge Regression?
            
            Es una variante de regresión lineal que incluye regularización bayesiana.
            
            **Ventajas:**
            - Maneja automáticamente la regularización
            - Proporciona estimaciones de incertidumbre
            - Robusto ante multicolinealidad
            - Menos propenso a overfitting que regresión lineal simple
            
            **Desventajas:**
            - Asume relaciones lineales
            - Más lento que regresión lineal simple
            - Puede no capturar relaciones no lineales
            
            **Cuándo usarlo:**
            - Relaciones aproximadamente lineales
            - Necesitas interpretabilidad de coeficientes
            - Datos con posible multicolinealidad
            
            **Alternativas:**
            - Regresión Lineal: Más simple, sin regularización
            - Ridge/Lasso: Regularización manual
            - Random Forest: Captura no linealidades
            - XGBoost: Rendimiento superior en muchos casos
            """)
        
        # Validación
        with st.spinner("Validando datos..."):
            es_valido, mensaje = validar_datos(df, x_col, y_col)
            if not es_valido:
                st.error(f"Error de validación: {mensaje}")
                return
        
        # Información del dataset
        st.subheader("Información del Dataset")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de filas", len(df))
        col2.metric("Variables predictoras", len(x_col))
        col3.metric("Tamaño de prueba", f"{test_size*100:.0f}%")
        col4.metric("CV Folds", cv_folds)
        
        # Manejo de valores nulos
        total_nulls = df[x_col + [y_col]].isnull().sum().sum()
        if total_nulls > 0:
            st.subheader("Manejo de Valores Nulos")
            null_info = df[x_col + [y_col]].isnull().sum()
            null_info = null_info[null_info > 0]
            st.warning(f"Se encontraron {total_nulls} valores nulos en {len(null_info)} columnas")
            
            col1, col2 = st.columns([2, 1])
            col1.dataframe(null_info.to_frame('Valores Nulos'), use_container_width=True)
            col2.info(f"**Estrategia:** {impute_strategy}")
            
            df = imputar_valores_faltantes(df, x_col, y_col, impute_strategy)
            st.success(f"Valores nulos imputados usando '{impute_strategy}'")
        
        # Detección de outliers
        st.subheader("Análisis de Outliers")
        df_original = df.copy()
        n_original = len(df_original)
        
        df_clean, outliers_mask, outliers_info = detectar_outliers(
            df, y_col, method=outlier_method, threshold=outlier_threshold
        )
        n_outliers = outliers_info['n_outliers']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de datos", n_original)
        col2.metric("Outliers detectados", n_outliers, 
                   delta=f"{(n_outliers/n_original*100):.1f}%" if n_original > 0 else "0%",
                   delta_color="inverse")
        
        if 'lower_bound' in outliers_info:
            col3.metric("Límite inferior", f"{outliers_info['lower_bound']:.2f}")
        if 'upper_bound' in outliers_info:
            col4.metric("Límite superior", f"{outliers_info['upper_bound']:.2f}")
        
        if n_outliers > 0:
            with st.expander("Ver outliers detectados", expanded=False):
                st.dataframe(outliers_info['outliers_df'], use_container_width=True)
                st.markdown(f"""
                **Método:** {outliers_info['method']}
                
                Estos valores están fuera del rango esperado y pueden ser:
                - Errores de captura de datos
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
        min_samples = max(10, len(x_col) * 5)
        if len(df) < min_samples:
            st.error(f"Datos insuficientes ({len(df)} observaciones). Se recomiendan al menos {min_samples}.")
            return
        
        # Preparación de datos
        X = df[x_col].values
        y = df[y_col].values
        y_min_original, y_max_original = y.min(), y.max()
        
        # Normalización
        scaler_X = seleccion_scaler(normalize_method)
        scaler_y = seleccion_scaler(normalize_method) if normalize_method != 'none' else None
        
        if normalize_method != 'none':
            X = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = y
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_scaled, test_size=test_size, random_state=random_state
        )
        
        # Entrenamiento
        with st.spinner("Entrenando modelo Bayesian Ridge..."):
            model = BayesianRidge(
                alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2,
                compute_score=True, fit_intercept=True, max_iter=300, tol=1e-3
            )
            model.fit(X_train, y_train)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Desnormalizar
        if normalize_method != 'none':
            y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
            y_pred_train_original = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
            y_pred_test_original = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
        else:
            y_train_original, y_test_original = y_train, y_test
            y_pred_train_original, y_pred_test_original = y_pred_train, y_pred_test
        
        # Métricas
        st.subheader("Métricas de Rendimiento")
        metrics_test = calcular_metricas(y_test_original, y_pred_test_original)
        metrics_train = calcular_metricas(y_train_original, y_pred_train_original)
        
        with st.expander("¿Qué significan estas métricas?", expanded=False):
            st.markdown("""
            **Guía de Métricas:**
            
            - **MSE/RMSE/MAE:** Miden el error promedio (menor es mejor)
            - **R²:** Proporción de varianza explicada (0-1, mayor es mejor)
            - **MAPE:** Error porcentual promedio
            - **Adjusted R²:** R² ajustado por número de variables
            - **Max/Min Error:** Rango de errores en las predicciones
            - **Std Error:** Variabilidad de los errores
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Conjunto de Prueba")
            st.dataframe(pd.DataFrame({
                "Métrica": list(metrics_test.keys()),
                "Valor": [f"{v:.4f}" for v in metrics_test.values()]
            }), use_container_width=True)
        
        with col2:
            st.markdown("### Conjunto de Entrenamiento")
            st.dataframe(pd.DataFrame({
                "Métrica": list(metrics_train.keys()),
                "Valor": [f"{v:.4f}" for v in metrics_train.values()]
            }), use_container_width=True)
        
        # Validación cruzada
        with st.spinner(f"Ejecutando validación cruzada con {cv_folds} folds..."):
            try:
                cv_scores = cross_val_score(
                    model, X, y_scaled, cv=cv_folds, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                cv_rmse = np.sqrt(-cv_scores)
            except Exception as e:
                st.warning(f"No se pudo completar la validación cruzada: {str(e)}")
        
        # Visualizaciones
        st.subheader("Visualizaciones de Predicciones")
        
        comparacion_df = pd.DataFrame({
            "Valor Real": y_test_original,
            "Valor Predicho": y_pred_test_original,
            "Error": y_test_original - y_pred_test_original,
            "Error Absoluto": np.abs(y_test_original - y_pred_test_original)
        })
        
        # Gráfico 1: Valores Reales vs Predichos
        st.markdown("#### Valores Reales vs Predichos")
        
        min_val = min(y_test_original.min(), y_pred_test_original.min())
        max_val = max(y_test_original.max(), y_pred_test_original.max())
        n_puntos = len(y_test_original)
        error_promedio = comparacion_df["Error Absoluto"].mean()
        
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown(f"""
            ### Gráfico de Dispersión: Valores Reales vs Predichos
            
            Este gráfico es fundamental para evaluar la **precisión** del modelo de regresión. Compara directamente 
            lo que el modelo predijo contra lo que realmente ocurrió en tus datos de prueba.
            
            ---
            
            #### Propósito del Gráfico
            
            Visualizar qué tan cerca están las predicciones del modelo de los valores reales. Si el modelo fuera 
            perfecto, todos los puntos caerían exactamente sobre la línea diagonal verde.
            
            ---
            
            #### Estructura del Gráfico
            
            **Eje Horizontal (X) - Valores Reales:**
            - Representa los valores verdaderos de la variable **'{y_col}'** que ya conoces
            - Rango actual: **{y_min_original:.2f}** hasta **{y_max_original:.2f}**
            - Este eje muestra la realidad de tus datos
            - Son los {n_puntos} valores del conjunto de prueba (datos que el modelo nunca vio durante el entrenamiento)
            
            **Eje Vertical (Y) - Valores Predichos:**
            - Representa lo que el modelo calculó para cada observación
            - Rango de predicciones: **{y_pred_test_original.min():.2f}** hasta **{y_pred_test_original.max():.2f}**
            - Estas son las estimaciones del modelo basadas en las variables predictoras
            
            **Línea Diagonal Verde (Predicción Perfecta):**
            - Es la línea de referencia donde X = Y
            - Representa el escenario ideal: si el modelo predijera exactamente el valor real
            - Ecuación: y = x (cada valor predicho = valor real)
            - **Puntos cercanos a esta línea** → El modelo está acertando
            - **Puntos alejados de esta línea** → El modelo está cometiendo errores
            
            **Puntos (Observaciones):**
            - Cada punto representa una observación del conjunto de prueba
            - Total de observaciones graficadas: **{n_puntos}**
            - **Color del punto** (escala de rojos):
            - Rojo claro/naranja → Error pequeño (buena predicción)
            - Rojo intenso/oscuro → Error grande (predicción deficiente)
            - **Tamaño del punto**:
            - Proporcional al error absoluto
            - Puntos más grandes = mayores errores
            - **Información al pasar el cursor**:
            - Valor real, valor predicho y error (diferencia entre ambos)
            
            ---
            
            #### Interpretación de Resultados Actuales
            
            **Rendimiento General:**
            - R² (coeficiente de determinación): **{metrics_test['R2']:.4f}**
            - Esto significa que el modelo explica el **{metrics_test['R2']*100:.1f}%** de la variabilidad en '{y_col}'
            - El **{100 - metrics_test['R2']*100:.1f}%** restante se debe a factores no capturados por el modelo
            
            - Error promedio absoluto: **{error_promedio:.4f}** unidades
            - En promedio, las predicciones se desvían {error_promedio:.4f} unidades del valor real
            - Esto representa un **{(error_promedio/(y_max_original - y_min_original)*100):.2f}%** del rango total de datos
            
            **Análisis de Dispersión:**
            - Error mínimo detectado: **{comparacion_df['Error Absoluto'].min():.4f}**
            - Error máximo detectado: **{comparacion_df['Error Absoluto'].max():.4f}**
            - Desviación estándar del error: **{metrics_test['Std_Error']:.4f}**
            
            ---
            
            #### Patrones Deseables (Modelo Confiable)
            
            1. **Alineación con la línea diagonal:**
            - Los puntos forman una nube estrecha alrededor de la línea verde
            - Indica que las predicciones son consistentemente precisas
            
            2. **Dispersión simétrica:**
            - Puntos distribuidos equitativamente arriba y abajo de la línea
            - No hay sesgo sistemático (ni sobre-estimación ni sub-estimación)
            
            3. **Densidad uniforme:**
            - La concentración de puntos es similar en todo el rango
            - El modelo predice bien tanto valores bajos como altos
            
            4. **Pocos puntos extremos:**
            - Colores mayormente claros (errores pequeños)
            - Pocos puntos rojos intensos alejados de la línea
            
            ---
            
            #### Problemas Potenciales a Detectar
            
            **1. Sub-estimación sistemática:**
            - La mayoría de puntos están **por encima** de la línea verde
            - Significa que los valores reales son mayores que las predicciones
            - El modelo está siendo **demasiado conservador**
            - Solución: revisar si faltan variables predictoras importantes
            
            **2. Sobre-estimación sistemática:**
            - La mayoría de puntos están **por debajo** de la línea verde
            - El modelo predice valores más altos que la realidad
            - Solución: simplificar el modelo o agregar regularización
            
            **3. Forma de abanico (heterocedasticidad):**
            - La dispersión aumenta hacia un extremo del gráfico
            - Indica que la **incertidumbre del modelo varía** según el valor predicho
            - Solución: transformar variables (logaritmo, raíz cuadrada) o usar modelos robustos
            
            **4. Outliers extremos:**
            - Puntos muy alejados (rojos intensos) de la línea diagonal
            - Representan casos que el modelo **no puede explicar**
            - Pueden ser:
            - Errores en la recolección de datos
            - Situaciones excepcionales no representadas en el entrenamiento
            - Límites del modelo (relaciones no lineales)
            
            **5. Agrupamientos o clusters:**
            - Puntos formando grupos separados
            - Sugiere la existencia de **subpoblaciones** no identificadas
            - Solución: agregar variables categóricas o de segmentación
            
            **6. Curvatura o patrón no lineal:**
            - Los puntos siguen una curva en lugar de línea recta
            - El modelo lineal **no captura la relación real** entre variables
            - Solución: agregar términos polinomiales o usar modelos no lineales
            
            ---
            
            #### Recomendaciones Prácticas
            
            **Si el modelo está funcionando bien (R² > 0.7, dispersión baja):**
            - El modelo está listo para hacer predicciones
            - Documenta los resultados y las variables utilizadas
            - Monitorea el rendimiento con datos nuevos
            
            **Si hay problemas (R² < 0.5, dispersión alta):**
            - Revisa los outliers
            - Considera agregar más variables predictoras relevantes
            - Explora transformaciones (log, potencias) si hay patrones curvos
            - Prueba modelos alternativos (Random Forest, XGBoost) para relaciones no lineales
            
            **Si hay sesgo evidente:**
            - Analiza si faltan variables importantes
            - Verifica la calidad de los datos de entrenamiento
            - Considera si las relaciones son realmente lineales
            
            ---
            
            #### Conceptos Clave
            
            - **Error = Valor Real - Valor Predicho**
            - Error positivo → modelo subestimó
            - Error negativo → modelo sobreestimó
            
            - **R² (0 a 1)**: Proporción de varianza explicada
            - 0.9-1.0: Excelente
            - 0.7-0.9: Bueno
            - 0.5-0.7: Moderado
            - < 0.5: Pobre (considera otro modelo)
            
            - **Error Absoluto Promedio**: Magnitud típica de error
            - Compáralo con el rango de tus datos para contexto
            - Ejemplo: error de 5 en rango de 10-100 es diferente a rango de 10-20
            """)
        
        fig1 = px.scatter(
            comparacion_df, x="Valor Real", y="Valor Predicho",
            color="Error Absoluto", size="Error Absoluto",
            hover_data=["Error"],
            title=f"Valores Reales vs Predichos ({n_puntos} observaciones)",
            color_continuous_scale="Reds",
            labels={
                "Valor Real": f"{y_col} (Real)",
                "Valor Predicho": f"{y_col} (Predicho)"
            }
        )
        
        fig1.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Predicción Perfecta (y=x)',
            line=dict(color='green', dash='dash', width=2)
        ))
        
        fig1.update_layout(annotations=[dict(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f'R² = {metrics_test["R2"]:.4f}<br>Error Promedio = {error_promedio:.4f}',
            showarrow=False, bgcolor='rgba(142, 174, 191,0.8)',
            borderwidth=1, borderpad=15
        )])
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico 2: Residuos vs Predichos
        st.markdown("#### Residuos vs Valores Predichos")
        
        residuos = comparacion_df["Error"]
        residuo_medio = residuos.mean()
        residuo_std = residuos.std()
        residuos_positivos = (residuos > 0).sum()
        residuos_negativos = (residuos < 0).sum()
        corr_residuos_predichos = np.corrcoef(np.abs(residuos), y_pred_test_original)[0, 1]
        
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown(f"""
            ### Gráfico de Residuos: Diagnóstico Estadístico del Modelo
            
            Este es el gráfico más importante para **validar los supuestos estadísticos** de tu modelo de regresión. 
            Los residuos son los errores que comete el modelo, y su comportamiento revela si el modelo es confiable 
            o tiene problemas estructurales.
            
            ---
            
            #### Propósito del Gráfico
            
            Detectar patrones en los errores del modelo que indiquen:
            - Violación de supuestos estadísticos
            - Relaciones no lineales no capturadas
            - Varianza no constante (heterocedasticidad)
            - Sesgo sistemático en las predicciones
            
            ---
            
            #### Estructura del Gráfico
            
            **Eje Horizontal (X) - Valores Predichos:**
            - Muestra las predicciones del modelo para '{y_col}'
            - Rango: **{y_pred_test_original.min():.2f}** hasta **{y_pred_test_original.max():.2f}**
            - Se usa el eje X para detectar si los errores cambian según el valor predicho
            - Representa el "nivel" o magnitud de la predicción
            
            **Eje Vertical (Y) - Residuos (Errores):**
            - **Residuo = Valor Real - Valor Predicho**
            - Rango actual: **{residuos.min():.2f}** hasta **{residuos.max():.2f}**
            - **Residuo positivo** (arriba de cero): El modelo **subestimó** (predijo menos de lo real)
            - **Residuo negativo** (debajo de cero): El modelo **sobreestimó** (predijo más de lo real)
            - **Residuo cercano a cero**: Predicción muy precisa
            
            **Línea Roja Horizontal (y = 0):**
            - Representa el error cero (predicción perfecta)
            - Es la referencia central del gráfico
            - Los puntos deben distribuirse aleatoriamente **alrededor** de esta línea
            - Si la mayoría de puntos está en un solo lado, hay sesgo sistemático
            
            **Bandas Sombreadas (Desviación Estándar):**
            
            **Banda Verde (±1 desviación estándar):**
            - Abarca desde **{residuo_medio - residuo_std:.2f}** hasta **{residuo_medio + residuo_std:.2f}**
            - En teoría, el **68%** de los puntos deberían estar dentro de esta banda
            - Porcentaje actual dentro de ±1σ: **{((residuos.abs() <= residuo_std).sum() / len(residuos) * 100):.1f}%**
            - Representa el rango "normal" de error esperado
            
            **Banda Amarilla (±2 desviaciones estándar):**
            - Abarca desde **{residuo_medio - 2*residuo_std:.2f}** hasta **{residuo_medio + 2*residuo_std:.2f}**
            - En teoría, el **95%** de los puntos deberían estar aquí
            - Porcentaje actual dentro de ±2σ: **{((residuos.abs() <= 2*residuo_std).sum() / len(residuos) * 100):.1f}%**
            - Puntos fuera de esta banda son outliers significativos
            
            **Puntos (Residuos Individuales):**
            - Cada punto representa el error de una predicción
            - Total de observaciones: **{n_puntos}**
            - Su posición vertical indica la magnitud y dirección del error
            
            ---
            
            #### Estadísticas Actuales de Residuos
            
            **Características de los Errores:**
            - **Media de residuos**: {residuo_medio:.4f}
            - Ideal: ≈ 0 (sin sesgo)
            - Tu valor: {'Excelente (sin sesgo)' if abs(residuo_medio) < 0.01 else 'Aceptable' if abs(residuo_medio) < residuo_std/2 else 'Hay sesgo sistemático'}
            
            - **Desviación estándar**: {residuo_std:.4f}
            - Mide la dispersión típica de los errores
            - Menor valor = predicciones más consistentes
            
            - **Distribución de residuos:**
            - Positivos (subestimación): {residuos_positivos} ({residuos_positivos/n_puntos*100:.1f}%)
            - Negativos (sobreestimación): {residuos_negativos} ({residuos_negativos/n_puntos*100:.1f}%)
            - Balance ideal: 50%-50%
            - Tu balance: {'Excelente (simétrico)' if abs(residuos_positivos/n_puntos - 0.5) < 0.05 else 'Aceptable' if abs(residuos_positivos/n_puntos - 0.5) < 0.15 else 'Desbalanceado'}
            
            - **Correlación residuos-predichos**: {corr_residuos_predichos:.4f}
            - Ideal: ≈ 0 (sin patrón)
            - Tu valor: {'Excelente (sin patrón)' if abs(corr_residuos_predichos) < 0.1 else 'Aceptable' if abs(corr_residuos_predichos) < 0.3 else 'Hay heterocedasticidad'}
            
            - **Rango de errores:**
            - Error más grande (positivo): {residuos.max():.4f} (subestimó por esta cantidad)
            - Error más grande (negativo): {residuos.min():.4f} (sobreestimó por esta cantidad)
            - Amplitud total: {residuos.max() - residuos.min():.4f}
            
            ---
            
            #### Patrón Ideal: Modelo Confiable
            
            Un modelo de regresión confiable debe mostrar:
            
            **1. Dispersión Aleatoria (Sin Patrón Visible):**
            - Los puntos no siguen ninguna forma geométrica
            - Parecen "esparcidos al azar" como confeti
            - No hay curvas, líneas ni tendencias
            - **Interpretación**: El modelo capturó todas las relaciones sistemáticas
            
            **2. Homocedasticidad (Varianza Constante):**
            - La "altura" de la nube de puntos es similar en todo el eje X
            - No se ensancha ni se estrecha al moverse de izquierda a derecha
            - La dispersión vertical es constante
            - **Interpretación**: La precisión del modelo no depende del valor predicho
            
            **3. Centrado en Cero:**
            - La media de los residuos es ≈ 0
            - Mitad de puntos arriba, mitad abajo de la línea roja
            - No hay desplazamiento sistemático
            - **Interpretación**: No hay sesgo (modelo balanceado)
            
            **4. Distribución Normal:**
            - La mayoría de puntos está cerca de y=0
            - Aproximadamente 68% dentro de ±1σ (banda verde)
            - Aproximadamente 95% dentro de ±2σ (banda amarilla)
            - Pocos outliers extremos
            - **Interpretación**: Los errores siguen una distribución gaussiana
            
            ---
            
            #### Patrones Problemáticos y Sus Causas
            
            **PROBLEMA 1: Heterocedasticidad (Forma de Embudo)**
            
            **Qué buscar:**
            - La dispersión aumenta o disminuye de izquierda a derecha
            - Forma de embudo, abanico o cono
            - Ejemplo: puntos apretados a la izquierda, dispersos a la derecha
            
            **Por qué ocurre:**
            - La variabilidad del error depende del valor predicho
            - Común cuando se modelan porcentajes, precios o tasas de crecimiento
            - Variables con rangos muy amplios sin transformar
            
            **Consecuencias:**
            - Los intervalos de confianza son incorrectos
            - Las pruebas estadísticas no son válidas
            - El modelo es menos confiable para ciertos rangos
            
            ---
            
            **PROBLEMA 2: Relación No Lineal (Patrón Curvo)**
            
            **Qué buscar:**
            - Los puntos forman una parábola (U o U invertida)
            - Patrón sinusoidal (onda)
            - Cualquier forma curva clara
            
            **Por qué ocurre:**
            - La relación real entre variables no es lineal
            - El modelo lineal intenta forzar una línea recta en datos curvos
            - Faltan términos cuadráticos, cúbicos o interacciones
            
            **Consecuencias:**
            - El modelo tiene sesgo sistemático
            - Subestima en ciertas regiones, sobreestima en otras
            - R² artificialmente bajo
            
            **Cómo detectarlo:**
            - Media de residuos positiva en los extremos, negativa en el centro (o viceversa)
            - Patrón visible en forma de U
            
            ---
            
            **PROBLEMA 3: Sesgo Sistemático (Línea Horizontal Desplazada)**
            
            **Qué buscar:**
            - La nube de puntos está mayormente arriba O abajo de y=0
            - Media de residuos significativamente diferente de cero
            - Desbalance: mucho más del 60% positivos o negativos
            
            **Por qué ocurre:**
            - El modelo consistentemente subestima o sobreestima
            - Intercepto mal calibrado
            - Falta una variable predictora importante
            - Los datos de entrenamiento no son representativos
            
            **Consecuencias:**
            - Predicciones siempre sesgadas en una dirección
            - Decisiones basadas en el modelo serán sistemáticamente erróneas
            
            **Cómo detectarlo:**
            - Media de residuos > 0.1 × desviación estándar
            - Tu media actual: **{residuo_medio:.4f}** ({abs(residuo_medio/residuo_std):.2f} × σ)
            - Balance actual: {residuos_positivos/n_puntos*100:.1f}% positivos

            ---
            
            **PROBLEMA 4: Outliers Extremos (Puntos Muy Alejados)**
            
            **Qué buscar:**
            - Puntos fuera de la banda amarilla (±2σ)
            - Residuos con valores absolutos muy grandes
            - Puntos aislados lejos del grupo principal
            
            **Por qué ocurre:**
            - Errores en recolección de datos
            - Eventos excepcionales no representados en el entrenamiento
            - El modelo no puede capturar casos extremos
            - Variables influyentes no incluidas
            
            **Consecuencias:**
            - Distorsionan las métricas de evaluación
            - Pueden indicar problemas graves de datos
            - Reducen la confianza en el modelo
            
            **Cómo detectarlo:**
            - Puntos fuera de ±2σ: **{(residuos.abs() > 2*residuo_std).sum()}** ({(residuos.abs() > 2*residuo_std).sum()/n_puntos*100:.1f}%)
            - Residuo máximo absoluto: **{residuos.abs().max():.4f}**
            - Esperado fuera de ±2σ: ~5% (teoría normal)
            
            ---
            
            **PROBLEMA 5: Agrupamientos o Bandas Horizontales**
            
            **Qué buscar:**
            - Puntos formando 2 o más líneas horizontales separadas
            - Clusters claramente diferenciados
            - "Escalones" en la distribución
            
            **Por qué ocurre:**
            - Existen subpoblaciones o categorías ocultas en los datos
            - Variable categórica importante no incluida en el modelo
            - Efectos de grupo no considerados
            
            **Consecuencias:**
            - El modelo trata casos diferentes como si fueran iguales
            - Predicciones imprecisas para ciertos grupos
            - R² artificialmente bajo
            
            ---
            
            **PROBLEMA 6: Tendencia Lineal (Correlación Alta)**
            
            **Qué buscar:**
            - Los residuos aumentan o disminuyen sistemáticamente con las predicciones
            - Pendiente visible en la nube de puntos
            - Correlación alta entre residuos y predicciones
            
            **Por qué ocurre:**
            - El modelo tiene sesgo proporcional
            - Problema de calibración del modelo
            - Falta ajustar la escala o intercepto
            
            **Consecuencias:**
            - Predicciones poco confiables en todo el rango
            - Indica problema fundamental en el modelo
            
            **Cómo detectarlo:**
            - Correlación |r| > 0.3
            - Tu correlación: **{corr_residuos_predichos:.4f}**
            ---
            
            #### Guía de Interpretación Rápida
            
            **TU MODELO ESTÁ BIEN SI:**
            - Los puntos parecen aleatoriamente dispersos
            - No ves ninguna forma clara (curva, embudo, línea)
            - Media ≈ 0 y correlación ≈ 0
            - Dispersión similar en todo el eje X
            - ~68% dentro de ±1σ, ~95% dentro de ±2σ
            
            **TU MODELO TIENE PROBLEMAS SI:**
            - Ves claramente algún patrón mencionado arriba
            - Correlación residuos-predichos > 0.3
            - Media > 0.1 × desviación estándar
            - Más del 10% de puntos fuera de ±2σ
            - Balance < 40% o > 60% en un lado
            
            **ANÁLISIS DE TU CASO:**
            
            Balance de residuos: {residuos_positivos/n_puntos*100:.1f}% positivos vs {residuos_negativos/n_puntos*100:.1f}% negativos
            - {'Excelente balance (simétrico)' if abs(residuos_positivos/n_puntos - 0.5) < 0.05 else 'Balance aceptable' if abs(residuos_positivos/n_puntos - 0.5) < 0.15 else 'Desbalance significativo - revisar sesgo'}
            
            Media de residuos: {residuo_medio:.4f} ({abs(residuo_medio/residuo_std):.2f} desviaciones estándar)
            - {'Sin sesgo sistemático' if abs(residuo_medio/residuo_std) < 0.1 else 'Sesgo aceptable' if abs(residuo_medio/residuo_std) < 0.3 else 'Sesgo significativo detectado'}
            
            Correlación residuos-predichos: {corr_residuos_predichos:.4f}
            - {'Sin patrón (excelente)' if abs(corr_residuos_predichos) < 0.1 else 'Patrón mínimo (aceptable)' if abs(corr_residuos_predichos) < 0.3 else 'Heterocedasticidad detectada'}
            
            Outliers (fuera de ±2σ): {(residuos.abs() > 2*residuo_std).sum()} ({(residuos.abs() > 2*residuo_std).sum()/n_puntos*100:.1f}%)
            - {'Cantidad normal (~5% esperado)' if (residuos.abs() > 2*residuo_std).sum()/n_puntos < 0.07 else 'Más outliers de lo esperado - revisar datos'}
            
            ---
            
            #### Conceptos Estadísticos Clave
            
            **Homocedasticidad:**
            - La varianza de los errores es constante
            - Supuesto fundamental de regresión lineal
            - Si se viola: usa regresión robusta o transformaciones
            
            **Normalidad de residuos:**
            - Los errores siguen distribución normal
            - Importante para inferencia estadística
            - Se verifica con histograma o Q-Q plot
            
            **Independencia:**
            - Los residuos no deben estar correlacionados entre sí
            - Este gráfico no lo verifica directamente
            - Importante en series temporales
            
            **Media cero:**
            - El error promedio debe ser cero
            - Garantiza que el modelo no está sesgado
            - Tu media: {residuo_medio:.4f}
            
            """)
        fig2 = px.scatter(
            x=y_pred_test_original, y=comparacion_df["Error"],
            title=f"Residuos vs Valores Predichos ({n_puntos} observaciones)",
            labels={"x": f"{y_col} (Predicho)", "y": "Residuo (Real - Predicho)"}
        )
        
        fig2.add_hline(y=0, line_dash="dash", line_color="red", line_width=2,
                      annotation_text="Error = 0")
        fig2.add_hrect(y0=residuo_medio - residuo_std, y1=residuo_medio + residuo_std,
                      fillcolor="green", opacity=0.1, line_width=0,
                      annotation_text="±1 std (68%)", annotation_position="right")
        fig2.add_hrect(y0=residuo_medio - 2*residuo_std, y1=residuo_medio + 2*residuo_std,
                      fillcolor="yellow", opacity=0.05, line_width=0,
                      annotation_text="±2 std (95%)", annotation_position="right")
        

        # 1. Limpia todas las anotaciones
        fig2.layout.annotations = ()

        # 2. Limpia textos en trazas
        fig2.update_traces(text=None, hovertext=None, hoverinfo='skip')

        # 3. Deshabilita hover y textos automáticos
        fig2.update_layout(hovermode=False)

        texto = (
                "Media = {0:.4f}<br>"
                "Std = {1:.4f}<br>"
                "Correlación = {2:.4f}"
        ).format(residuo_medio, residuo_std, corr_residuos_predichos)
        
        fig2.update_layout(annotations=[dict(
            x=0.01, y=0.98, xref='paper', yref='paper',
            text=texto,
            showarrow=False, bgcolor='rgba(142, 174, 191,0.8)',
            borderwidth=1,borderpad=12,  align="left"
        )])
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Análisis de convergencia
        if hasattr(model, 'scores_') and len(model.scores_) > 0:
            st.subheader("Análisis de Convergencia del Modelo")
            
            n_iteraciones = len(model.scores_)
            score_inicial = model.scores_[0]
            score_final = model.scores_[-1]
            mejora_total = score_final - score_inicial
            mejora_porcentual = (mejora_total / abs(score_inicial) * 100) if score_inicial != 0 else 0
            
            ultimas_5_mejoras = [model.scores_[i] - model.scores_[i-1] 
                                for i in range(-5, 0)] if n_iteraciones >= 10 else []
            mejora_promedio_final = np.mean(ultimas_5_mejoras) if ultimas_5_mejoras else 0
            
            with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
                st.markdown(f"""
                ### Gráfico de Convergencia: Proceso de Optimización del Modelo
                
                Este gráfico muestra cómo el modelo Bayesian Ridge mejora iterativamente durante el proceso de entrenamiento. 
                Es una ventana al "aprendizaje" del algoritmo y permite verificar que el entrenamiento fue exitoso.
                
                ---
                
                #### Propósito del Gráfico
                
                Visualizar el proceso de optimización para:
                - Confirmar que el modelo alcanzó la mejor solución posible
                - Detectar problemas en el entrenamiento (no convergencia, inestabilidad)
                - Decidir si se necesitan más iteraciones
                - Validar que los hiperparámetros son adecuados
                
                ---
                
                #### Estructura del Gráfico
                
                **Eje Horizontal (X) - Iteraciones:**
                - Representa cada paso del algoritmo de optimización
                - Rango: 0 hasta **{n_iteraciones}** iteraciones
                - Cada punto es una actualización de los parámetros del modelo
                - El algoritmo ajusta los coeficientes paso a paso buscando el mejor ajuste
                - Límite máximo configurado: 300 iteraciones
                
                **Eje Vertical (Y) - Score (Log-verosimilitud Marginal):**
                - Medida de la calidad del ajuste del modelo
                - Valor inicial (iteración 0): **{score_inicial:.4f}**
                - Valor final (iteración {n_iteraciones-1}): **{score_final:.4f}**
                - **Mejora total**: {mejora_total:.4f} ({mejora_porcentual:+.2f}%)
                - **Valores más altos** = Mejor ajuste del modelo a los datos
                - Esta métrica es específica del método bayesiano (no es R² ni MSE)
                
                **Línea Azul (Evolución del Score):**
                - Conecta los valores del score en cada iteración
                - Muestra la "trayectoria" del aprendizaje
                - **Pendiente ascendente** = El modelo está mejorando
                - **Línea horizontal** = El modelo se estabilizó (convergió)
                - **Oscilaciones** = Posible inestabilidad en el entrenamiento
                
                **Línea Roja Discontinua (Tendencia Últimas 10 Iteraciones):**
                - Ajuste lineal de los últimos 10 puntos
                - Permite ver si aún hay mejora al final del entrenamiento
                - **Pendiente casi horizontal** = Convergencia exitosa
                - **Pendiente ascendente** = Aún está mejorando (necesita más iteraciones)
                - Mejora promedio últimas 5 iteraciones: **{mejora_promedio_final:.6f}**
                
                **Anotaciones (Inicio y Final):**
                - Marcan el score en la primera y última iteración
                - Permiten cuantificar la mejora total
                - Facilitan comparar el punto de partida con el resultado final
                
                ---
                
                #### Análisis del Proceso de Entrenamiento Actual
                
                **Rendimiento del Algoritmo:**
                - **Total de iteraciones ejecutadas**: {n_iteraciones}
                - **Iteraciones disponibles**: 300 (límite máximo)
                - **Uso del presupuesto**: {n_iteraciones/300*100:.1f}% de iteraciones utilizadas
                
                **Evolución del Score:**
                - **Punto de partida** (iteración 0): {score_inicial:.4f}
                - **Punto final** (iteración {n_iteraciones-1}): {score_final:.4f}
                - **Mejora absoluta**: {mejora_total:.4f} unidades
                - **Mejora relativa**: {mejora_porcentual:+.2f}%
                
                **Convergencia:**
                - **Mejora promedio últimas 5 iteraciones**: {mejora_promedio_final:.6f}
                - **Estado**: {'Convergió exitosamente' if n_iteraciones < 300 and mejora_promedio_final < 0.001 
                else 'Alcanzó el límite - considera aumentar max_iter' if n_iteraciones >= 300 
                else 'En proceso de convergencia'}
                
                **Velocidad de Convergencia:**
                - **Fase inicial** (primeras 20% iteraciones): Mejora rápida esperada
                - **Fase media** (20%-80%): Mejora moderada
                - **Fase final** (últimas 20%): Refinamiento fino
                
                ---
                
                #### Patrones de Convergencia Exitosa
                
                **1. Curva de Aprendizaje Típica:**
                - **Inicio empinado**: Mejora rápida en las primeras 10-50 iteraciones
                - **Transición gradual**: La pendiente disminuye progresivamente
                - **Meseta final**: Se estabiliza en un valor (línea casi horizontal)
                - **Interpretación**: El algoritmo encontró la solución óptima eficientemente
                
                **2. Estabilización Temprana (Convergencia Rápida):**
                - El score se estabiliza antes de llegar a 300 iteraciones
                - Mejora promedio últimas iteraciones < 0.001
                - Tu caso: {'SÍ' if mejora_promedio_final < 0.001 else 'NO'}
                - **Interpretación**: 
                - Positivo: Algoritmo eficiente, problema bien condicionado
                - El modelo encontró la solución en {n_iteraciones} iteraciones
                - No se necesitan más iteraciones
                
                **3. Mejora Monotónica:**
                - El score siempre aumenta (nunca disminuye)
                - No hay oscilaciones bruscas
                - Progreso constante y predecible
                - **Interpretación**: Optimización estable y confiable
                
                **4. Asíntota Clara:**
                - La línea se aplana completamente al final
                - Las últimas 20-50 iteraciones prácticamente no mejoran el score
                - **Interpretación**: Se alcanzó el límite teórico del modelo
                
                ---
                
                #### Problemas de Convergencia a Detectar
                
                **PROBLEMA 1: No Convergió (Alcanzó el Límite)**
                
                **Qué buscar:**
                - Llegó a las 300 iteraciones (límite máximo)
                - La línea roja de tendencia sigue ascendente al final
                - Mejora promedio últimas 5 iteraciones > 0.001
                - Tu caso: {'SÍ - Alcanzó límite' if n_iteraciones >= 300 else 'NO'}
                
                **Por qué ocurre:**
                - El problema es complejo (muchas variables, relaciones complicadas)
                - Los hiperparámetros están mal ajustados
                - El tolerance (tol) es demasiado estricto
                - La escala de los datos dificulta la optimización
                
                **Consecuencias:**
                - El modelo podría mejorar con más iteraciones
                - Puede no ser la mejor solución posible
                - Rendimiento subóptimo
                
                ---
                
                **PROBLEMA 2: Convergencia Prematura**
                
                **Qué buscar:**
                - Se estabiliza en menos de 10-20 iteraciones
                - El score final es sorprendentemente bajo
                - La mejora total es mínima (< 1%)
                - No hay una curva de aprendizaje clara
                
                **Por qué ocurre:**
                - Hiperparámetros demasiado restrictivos (regularización muy fuerte)
                - Datos muy simples (relación casi perfecta)
                - Problema en la inicialización
                - Tolerance demasiado relajado
                
                **Consecuencias:**
                - El modelo se "rinde" demasiado pronto
                - Puede haber dejado mejora sobre la mesa
                - Rendimiento inferior al potencial
                
                ---
                
                **PROBLEMA 3: Oscilaciones o Inestabilidad**
                
                **Qué buscar:**
                - La línea azul sube y baja repetidamente
                - No hay progreso suave
                - Cambios bruscos en el score entre iteraciones
                - Patrón en "zigzag" o "dientes de sierra"
                
                **Por qué ocurre:**
                - Learning rate implícito muy alto
                - Datos mal escalados (rangos muy diferentes entre variables)
                - Multicolinealidad severa
                - Presencia de outliers extremos
                - Hiperparámetros inadecuados
                
                **Consecuencias:**
                - El algoritmo no encuentra una dirección clara
                - Puede no converger nunca
                - Solución inestable y poco confiable
                
                ---
                
                **PROBLEMA 4: Estancamiento Intermedio**
                
                **Qué buscar:**
                - Mejora rápida al inicio
                - Meseta prolongada en medio del entrenamiento
                - Luego otra mejora o no más progreso
                - Patrón de "escalones"
                
                **Por qué ocurre:**
                - El algoritmo encontró un mínimo local
                - Dificultad para escapar de cierta configuración
                - Geometría compleja del espacio de soluciones
                
                **Consecuencias:**
                - Puede no ser la mejor solución global
                - Tiempo desperdiciado sin mejora
                
                ---
                
                **PROBLEMA 5: Mejora Negativa (Score Disminuye)**
                
                **Qué buscar:**
                - El score baja en algún momento
                - Retroceso respecto a iteraciones anteriores
                - **MUY RARO** en Bayesian Ridge (indica problema serio)
                
                **Por qué ocurre:**
                - Error numérico grave
                - Datos corruptos
                - Bug en la implementación
                - Problema de precisión floating-point
                
                **Consecuencias:**
                - El modelo no es confiable
                - Resultados impredecibles
                
                ---
                
                #### Interpretación de Tu Gráfico Actual
                
                **Análisis del Comportamiento:**
                
                **Fase Inicial (primeras {int(n_iteraciones*0.2)} iteraciones):**
                - {'Mejora rápida esperada' if n_iteraciones > 10 else 'Muy pocas iteraciones para evaluar'}
                - El algoritmo explora rápidamente el espacio de soluciones
                
                **Fase de Convergencia (últimas {int(n_iteraciones*0.2)} iteraciones):**
                - Mejora promedio: {mejora_promedio_final:.6f}
                - {'Convergió exitosamente (mejora < 0.001)' if mejora_promedio_final < 0.001 
                else 'Aún mejorando significativamente (mejora > 0.001)' if mejora_promedio_final > 0.01
                else 'En proceso de convergencia (0.001 < mejora < 0.01)'}
                
                **Estado Final:**
                - Score final: {score_final:.4f}
                - {'Entrenamiento completo y exitoso' if n_iteraciones < 300 and mejora_promedio_final < 0.001
                else 'Considera aumentar max_iter a 500 o más' if n_iteraciones >= 300
                else 'Convergencia en progreso, aceptable'}
                
                ---
                
                #### Recomendaciones Específicas para Tu Caso
                
                **Basado en {n_iteraciones} iteraciones y mejora de {mejora_promedio_final:.6f}:**
                
                {'''
                **TU MODELO ESTÁ BIEN ENTRENADO:**
                - Convergió exitosamente antes del límite
                - La mejora final es insignificante
                - El modelo alcanzó su máximo potencial
                - No necesitas hacer cambios
                - Puedes confiar en las predicciones
                ''' if n_iteraciones < 300 and mejora_promedio_final < 0.001 else ''}
                
                {'''
                **CONSIDERA AUMENTAR ITERACIONES:**
                - Alcanzaste el límite de 300 iteraciones
                - Aún hay mejora significativa al final
                - Recomendación: Cambiar max_iter a 500 o 1000
                - Esto puede mejorar el rendimiento del modelo
                - Monitorea si la mejora continúa
                ''' if n_iteraciones >= 300 and mejora_promedio_final > 0.001 else ''}
                
                {'''
                **CONVERGENCIA ACEPTABLE:**
                - Aunque alcanzó el límite, ya convergió
                - La mejora final es mínima
                - El modelo está bien optimizado
                - Opcional: Aumentar max_iter para confirmar
                - El rendimiento actual es confiable
                ''' if n_iteraciones >= 300 and mejora_promedio_final < 0.001 else ''}
                
                {'''
                **EN PROCESO DE CONVERGENCIA:**
                - No alcanzó el límite todavía
                - Aún hay mejora moderada
                - El algoritmo está trabajando correctamente
                - Espera a que se estabilice naturalmente
                - Si llegas a 300, considera aumentar max_iter
                ''' if n_iteraciones < 300 and mejora_promedio_final >= 0.001 else ''}
                """)
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=list(range(n_iteraciones)), y=model.scores_,
                mode='lines+markers', name='Score del Modelo',
                line=dict(color='blue', width=2), marker=dict(size=6)
            ))
            
            if n_iteraciones >= 10:
                x_ultimas = list(range(n_iteraciones-10, n_iteraciones))
                y_ultimas = model.scores_[-10:]
                z = np.polyfit(x_ultimas, y_ultimas, 1)
                p = np.poly1d(z)
                fig3.add_trace(go.Scatter(
                    x=x_ultimas, y=p(x_ultimas), mode='lines',
                    name='Tendencia (últimas 10)',
                    line=dict(color='red', dash='dash', width=2)
                ))
            
            fig3.update_layout(
                title=f"Evolución del Score ({n_iteraciones} iteraciones)",
                xaxis_title="Iteración",
                yaxis_title="Score (Log-verosimilitud)",
                hovermode='x unified',
                annotations=[
                    dict(x=0, y=score_inicial, text=f"Inicio: {score_inicial:.4f}",
                         showarrow=True, arrowhead=3, bgcolor='rgba(142, 174, 191,0.8)', borderpad=8),
                    dict(x=n_iteraciones-1, y=score_final, text=f"Final: {score_final:.4f}",
                         showarrow=True, arrowhead=3, bgcolor='rgba(142, 174, 191,0.8)', borderpad=8)
                ]
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            if n_iteraciones >= 300:
                st.warning(f"El modelo alcanzó el máximo de iteraciones (300). Mejora promedio últimas 5: {mejora_promedio_final:.6f}")
            elif mejora_promedio_final < 1e-6:
                st.success(f"Convergencia exitosa en {n_iteraciones} iteraciones")
        
        # Comparación Train vs Test
        st.subheader("Comparación Train vs Test")
        
        diff_r2 = metrics_train['R2'] - metrics_test['R2']
        diff_rmse = metrics_test['RMSE'] - metrics_train['RMSE']
        diff_mae = metrics_test['MAE'] - metrics_train['MAE']

        if diff_r2 < 0.05:
            color_diag = "verde"
        elif diff_r2 < 0.10:
                color_diag = "amarillo"
        elif diff_r2 < 0.20:
                color_diag = "naranja"
        else:
            color_diag = "rojo"

        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
                st.markdown(f"""
                ### Comparación Train vs Test: Evaluación de Generalización del Modelo
                
                Este es el gráfico definitivo para detectar **overfitting** (sobreajuste). Compara el rendimiento del modelo 
                en los datos que usó para aprender versus datos completamente nuevos que nunca vio. La diferencia entre 
                ambos conjuntos revela si tu modelo es realmente útil o solo memorizó los datos de entrenamiento.
                
                ---
                
                #### Propósito del Gráfico
                
                Responder la pregunta crítica: **¿El modelo realmente aprendió patrones generales o solo memorizó casos específicos?**
                
                - Detectar overfitting (modelo memoriza en lugar de aprender)
                - Detectar underfitting (modelo demasiado simple)
                - Validar que el modelo es útil para datos nuevos
                - Decidir si el modelo está listo para producción
                
                ---
                
                #### Estructura del Gráfico
                
                **Barras Agrupadas por Métrica:**
                El gráfico muestra tres métricas clave, cada una con dos barras:
                
                **1. R² (Coeficiente de Determinación):**
                - **Barra azul (Entrenamiento)**: {metrics_train['R2']:.4f}
                - **Barra naranja (Prueba)**: {metrics_test['R2']:.4f}
                - **Diferencia**: {diff_r2:.4f}
                - **Qué mide**: Proporción de varianza explicada por el modelo
                - **Escala**: -∞ a 1.0 (1.0 = perfecto)
                - **Interpretación**: Mayor = mejor
                - Esta métrica responde: "¿Qué tan bien el modelo explica la variabilidad de los datos?"
                
                **2. RMSE (Root Mean Squared Error):**
                - **Barra verde (Entrenamiento)**: {metrics_train['RMSE']:.4f}
                - **Barra roja (Prueba)**: {metrics_test['RMSE']:.4f}
                - **Diferencia**: {diff_rmse:+.4f} ({diff_rmse/metrics_train['RMSE']*100:+.1f}%)
                - **Qué mide**: Error promedio con más peso a errores grandes
                - **Unidades**: Mismas que '{y_col}'
                - **Interpretación**: Menor = mejor
                - Penaliza fuertemente las predicciones muy equivocadas
                
                **3. MAE (Mean Absolute Error):**
                - **Barra morada (Entrenamiento)**: {metrics_train['MAE']:.4f}
                - **Barra marrón (Prueba)**: {metrics_test['MAE']:.4f}
                - **Diferencia**: {diff_mae:+.4f} ({diff_mae/metrics_train['MAE']*100:+.1f}%)
                - **Qué mide**: Error promedio absoluto sin ponderación
                - **Unidades**: Mismas que '{y_col}'
                - **Interpretación**: Menor = mejor
                - Más robusto a outliers que RMSE
                
                **División de Datos:**
                - **Conjunto de Entrenamiento**: {len(y_train_original)} observaciones ({100-test_size*100:.0f}%)
                - Datos que el modelo usó para aprender
                - El modelo "vio" estos valores durante el entrenamiento
                
                - **Conjunto de Prueba**: {len(y_test_original)} observaciones ({test_size*100:.0f}%)
                - Datos completamente nuevos para el modelo
                - El modelo NUNCA los vio durante el entrenamiento
                - Simulan datos del mundo real cuando uses el modelo
                
                **Código de Colores:**
                - **Colores fríos (azul, verde, morado)**: Entrenamiento
                - **Colores cálidos (naranja, rojo, marrón)**: Prueba
                - Facilita identificar visualmente qué barra pertenece a qué conjunto
                
                **Anotación Superior (Diagnóstico):**
                - Muestra ΔR² (diferencia en R²)
                - **Color verde**: Generalización buena
                - **Color naranja**: Overfitting leve
                - **Color rojo**: Overfitting significativo
                - Tu caso: **ΔR² = {diff_r2:.4f}** - {color_diag.upper()}
                
                ---
                
                #### Análisis Detallado de Resultados
                
                **MÉTRICA 1: R² (Varianza Explicada)**
                
                **¿Qué es R²?**
                - Proporción de variabilidad en '{y_col}' que el modelo puede explicar
                - R² = 1.0 → Modelo perfecto (explica 100% de la varianza)
                - R² = 0.0 → Modelo inútil (no mejor que predecir la media)
                - R² < 0.0 → Modelo peor que predecir siempre la media
                
                **Tus valores:**
                - **R² Entrenamiento**: {metrics_train['R2']:.4f} ({metrics_train['R2']*100:.2f}% varianza explicada)
                - {' Excelente' if metrics_train['R2'] > 0.9 else 'Bueno' if metrics_train['R2'] > 0.7 else 'Moderado' if metrics_train['R2'] > 0.5 else 'Pobre'}
                - **R² Prueba**: {metrics_test['R2']:.4f} ({metrics_test['R2']*100:.2f}% varianza explicada)
                - {'Excelente' if metrics_test['R2'] > 0.9 else 'Bueno' if metrics_test['R2'] > 0.7 else 'Moderado' if metrics_test['R2'] > 0.5 else 'Pobre'}
                
                **Gap de Generalización:**
                - **ΔR²** (Train - Test): {diff_r2:.4f}
                - Interpretación: El modelo explica un {diff_r2*100:.2f}% menos de varianza en datos nuevos
                - {'Excelente generalización' if diff_r2 < 0.05 else 'Buena generalización' if diff_r2 < 0.10 else 'Overfitting leve' if diff_r2 < 0.15 else 'Overfitting moderado' if diff_r2 < 0.25 else 'Overfitting severo'}
                
                **¿Por qué Train es mejor?**
                - El modelo ajustó sus parámetros específicamente para minimizar errores en Train
                - Es como estudiar con las respuestas del examen → mejor rendimiento en esas preguntas
                - La pregunta clave es: ¿CUÁNTO mejor es Train?
                
                ---
                
                **MÉTRICA 2: RMSE (Error Cuadrático Medio)**
                
                **¿Qué es RMSE?**
                - Raíz del promedio de errores al cuadrado
                - Penaliza fuertemente errores grandes (por el cuadrado)
                - Si un error es 2x más grande, contribuye 4x más al RMSE
                - Útil cuando errores grandes son especialmente problemáticos
                
                **Tus valores:**
                - **RMSE Entrenamiento**: {metrics_train['RMSE']:.4f}
                - **RMSE Prueba**: {metrics_test['RMSE']:.4f}
                - **Diferencia**: {diff_rmse:+.4f} ({diff_rmse/metrics_train['RMSE']*100:+.1f}%)
                
                **Ratio RMSE (Test/Train):**
                - **Ratio**: {metrics_test['RMSE']/metrics_train['RMSE']:.2f}x
                - Interpretación: El error en prueba es {metrics_test['RMSE']/metrics_train['RMSE']:.2f} veces el error en entrenamiento
                - {'Excelente (< 1.2x)' if metrics_test['RMSE']/metrics_train['RMSE'] < 1.2 else 'Bueno (1.2-1.5x)' if metrics_test['RMSE']/metrics_train['RMSE'] < 1.5 else 'Overfitting (1.5-2.0x)' if metrics_test['RMSE']/metrics_train['RMSE'] < 2.0 else 'Overfitting severo (> 2.0x)'}
                
                **Contexto del Error:**
                - Rango de '{y_col}': {y_min_original:.2f} a {y_max_original:.2f}
                - Amplitud total: {y_max_original - y_min_original:.2f}
                - RMSE como % del rango: {metrics_test['RMSE']/(y_max_original - y_min_original)*100:.2f}%
                - {'Error bajo (< 10% del rango)' if metrics_test['RMSE']/(y_max_original - y_min_original) < 0.10 else 'Error moderado (10-20%)' if metrics_test['RMSE']/(y_max_original - y_min_original) < 0.20 else 'Error alto (> 20%)'}
                
                ---
                
                **MÉTRICA 3: MAE (Error Absoluto Medio)**
                
                **¿Qué es MAE?**
                - Promedio simple de errores absolutos
                - Todos los errores cuentan igual (no penaliza más los grandes)
                - Más interpretable: "en promedio, el modelo se equivoca por X unidades"
                - Más robusto a outliers que RMSE
                
                **Tus valores:**
                - **MAE Entrenamiento**: {metrics_train['MAE']:.4f}
                - **MAE Prueba**: {metrics_test['MAE']:.4f}
                - **Diferencia**: {diff_mae:+.4f} ({diff_mae/metrics_train['MAE']*100:+.1f}%)
                
                **Ratio MAE (Test/Train):**
                - **Ratio**: {metrics_test['MAE']/metrics_train['MAE']:.2f}x
                - {'Excelente (< 1.2x)' if metrics_test['MAE']/metrics_train['MAE'] < 1.2 else 'Bueno (1.2-1.5x)' if metrics_test['MAE']/metrics_train['MAE'] < 1.5 else 'Overfitting (1.5-2.0x)' if metrics_test['MAE']/metrics_train['MAE'] < 2.0 else 'Overfitting severo (> 2.0x)'}
                
                **Comparación RMSE vs MAE:**
                - Si RMSE >> MAE: Hay outliers importantes (errores muy grandes ocasionales)
                - Si RMSE ≈ MAE: Errores consistentes sin casos extremos
                - Tu caso: RMSE/MAE ratio = {metrics_test['RMSE']/metrics_test['MAE']:.2f}
                - {'Errores consistentes' if metrics_test['RMSE']/metrics_test['MAE'] < 1.3 else 'Presencia de outliers significativos'}
                
                ---
                
                #### Escenarios de Generalización (¿Tu Modelo es Útil?)
                
                **ESCENARIO 1: GENERALIZACIÓN EXCELENTE**
                
                **Características:**
                - ΔR² < 0.05 (diferencia muy pequeña)
                - Ratio RMSE < 1.15x
                - Ratio MAE < 1.15x
                - Ambos conjuntos tienen buen rendimiento
                
                **Interpretación:**
                - El modelo aprendió patrones reales y generales
                - NO memorizó datos específicos
                - Rendimiento similar en datos nuevos y conocidos
                - **El modelo es confiable y listo para usar**
                
                **Tu caso:**
                - ΔR²: {diff_r2:.4f} {'' if diff_r2 < 0.05 else ''}
                - Ratio RMSE: {metrics_test['RMSE']/metrics_train['RMSE']:.2f}x {'' if metrics_test['RMSE']/metrics_train['RMSE'] < 1.15 else ''}
                - {' TU MODELO CAE EN ESTE ESCENARIO' if diff_r2 < 0.05 and metrics_test['RMSE']/metrics_train['RMSE'] < 1.15 else ''}
                
                ---
                
                **ESCENARIO 2: GENERALIZACIÓN BUENA**
                
                **Características:**
                - 0.05 < ΔR² < 0.10
                - 1.15x < Ratio RMSE < 1.5x
                - 1.15x < Ratio MAE < 1.5x
                - Degradación aceptable en datos nuevos
                
                **Interpretación:**
                - El modelo generalmente aprende bien
                - Pequeña ventaja en datos de entrenamiento (esperado)
                - Rendimiento aceptable en datos nuevos
                - **El modelo es útil con precauciones**
                
                **Tu caso:**
                - ΔR²: {diff_r2:.4f} {'' if 0.05 <= diff_r2 < 0.10 else ''}
                - Ratio RMSE: {metrics_test['RMSE']/metrics_train['RMSE']:.2f}x {'' if 1.15 <= metrics_test['RMSE']/metrics_train['RMSE'] < 1.5 else ''}
                - {' TU MODELO CAE EN ESTE ESCENARIO' if 0.05 <= diff_r2 < 0.10 and 1.15 <= metrics_test['RMSE']/metrics_train['RMSE'] < 1.5 else ''}
                
                ---
                
                **ESCENARIO 3: OVERFITTING LEVE**
                
                **Características:**
                - 0.10 < ΔR² < 0.20
                - 1.5x < Ratio RMSE < 2.0x
                - Train tiene rendimiento notablemente mejor
                - Test tiene rendimiento aceptable pero degradado
                
                **Interpretación:**
                - El modelo memorizó algunos patrones específicos de Train
                - Aún captura tendencias generales
                - Degradación significativa en datos nuevos
                - **El modelo es cuestionable**
                
                **Tu caso:**
                - ΔR²: {diff_r2:.4f} {'' if 0.10 <= diff_r2 < 0.20 else ''}
                - Ratio RMSE: {metrics_test['RMSE']/metrics_train['RMSE']:.2f}x {'' if 1.5 <= metrics_test['RMSE']/metrics_train['RMSE'] < 2.0 else ''}
                - {' TU MODELO CAE EN ESTE ESCENARIO' if 0.10 <= diff_r2 < 0.20 or (1.5 <= metrics_test['RMSE']/metrics_train['RMSE'] < 2.0) else ''}
                
                **Causas comunes:**
                - Insuficientes datos de entrenamiento
                - Modelo muy complejo para los datos disponibles
                - Variables irrelevantes incluidas
                - Falta de regularización

                ---
                
                **ESCENARIO 4: OVERFITTING SEVERO**
                
                **Características:**
                - ΔR² > 0.20
                - Ratio RMSE > 2.0x
                - Train casi perfecto (R² > 0.95)
                - Test pobre (R² < 0.70)
                - Diferencia dramática entre conjuntos
                
                **Interpretación:**
                - El modelo está **memorizando** en lugar de aprender
                - Captura ruido y casos específicos, no patrones generales
                - **Completamente inútil para datos nuevos**
                
                **Tu caso:**
                - ΔR²: {diff_r2:.4f} {'' if diff_r2 > 0.20 else ''}
                - Ratio RMSE: {metrics_test['RMSE']/metrics_train['RMSE']:.2f}x {'' if metrics_test['RMSE']/metrics_train['RMSE'] > 2.0 else ''}
                - {' TU MODELO CAE EN ESTE ESCENARIO - CRÍTICO' if diff_r2 > 0.20 or metrics_test['RMSE']/metrics_train['RMSE'] > 2.0 else ''}
                
                **Causas comunes:**
                - Muy pocos datos (<100 observaciones)
                - Demasiadas variables (curse of dimensionality)
                - Modelo extremadamente complejo
                - Datos de entrenamiento no representativos
                
                **Qué hacer - URGENTE:**
                - **NO USES ESTE MODELO** - No es confiable
                - Recolectar significativamente más datos (3-10x más)
                - Reducir drásticamente número de variables
                - Usar modelo más simple (regresión lineal simple)
                - Aplicar regularización muy fuerte
                - Considerar si el problema es solucionable con los datos disponibles
                
                ---
                
                **ESCENARIO 5: UNDERFITTING (Ambos Conjuntos Pobres)**
                
                **Características:**
                - R² < 0.50 en ambos conjuntos
                - RMSE alto en ambos
                - ΔR² pequeño (modelo consistentemente malo)
                - No mejora en ningún conjunto
                
                **Interpretación:**
                - El modelo es **demasiado simple** para capturar los patrones
                - O las variables predictoras no son útiles
                - O la relación no es lineal
                - **El modelo no aprendió nada útil**
                
                **Tu caso:**
                - R² Test: {metrics_test['R2']:.4f} {'' if metrics_test['R2'] < 0.50 else ''}
                - R² Train: {metrics_train['R2']:.4f} {'' if metrics_train['R2'] < 0.50 else ''}
                - {' TU MODELO CAE EN ESTE ESCENARIO' if metrics_test['R2'] < 0.50 and metrics_train['R2'] < 0.50 else ''}
                
                **Causas comunes:**
                - Variables predictoras irrelevantes o débiles
                - Relación no lineal entre variables
                - Datos insuficientes o de mala calidad
                - Problema inherentemente difícil de predecir
                
                ---
                
                #### Diagnóstico Final de Tu Modelo
                
                **Resumen Ejecutivo:**
                
                **Rendimiento Test (Lo Que Importa):**
                - R² = {metrics_test['R2']:.4f} → {'Excelente' if metrics_test['R2'] > 0.9 else 'Bueno' if metrics_test['R2'] > 0.7 else 'Moderado' if metrics_test['R2'] > 0.5 else 'Pobre'}
                - RMSE = {metrics_test['RMSE']:.4f} ({metrics_test['RMSE']/(y_max_original - y_min_original)*100:.1f}% del rango)
                - MAE = {metrics_test['MAE']:.4f}
                
                **Gap de Generalización:**
                - ΔR² = {diff_r2:.4f} → {'Sin overfitting' if diff_r2 < 0.10 else 'Overfitting leve' if diff_r2 < 0.15 else 'Overfitting moderado' if diff_r2 < 0.25 else 'Overfitting severo'}
                - Ratio RMSE = {metrics_test['RMSE']/metrics_train['RMSE']:.2f}x → {'Excelente' if metrics_test['RMSE']/metrics_train['RMSE'] < 1.2 else 'Bueno' if metrics_test['RMSE']/metrics_train['RMSE'] < 1.5 else 'Overfitting'}
                
                **Veredicto:**
                {'''
                **MODELO APROBADO - LISTO PARA USAR**
                Tu modelo generaliza excelentemente. El rendimiento en datos nuevos es similar al de entrenamiento,
                lo que indica que aprendió patrones reales y no memorizó casos específicos.
                ''' if diff_r2 < 0.10 and metrics_test['R2'] > 0.7 else ''}
                
                {'''
                **MODELO CUESTIONABLE - MEJORAR ANTES DE USAR**
                Hay evidencia de overfitting. El modelo funciona mejor en datos conocidos que en datos nuevos.
                Recomendación: Implementar las mejoras sugeridas arriba antes de usar en producción.
                ''' if (0.10 <= diff_r2 < 0.25 and metrics_test['R2'] > 0.5) else ''}
                
                {'''
                **MODELO NO APROBADO - NO USAR**
                Overfitting severo detectado. El modelo es inútil para datos nuevos. Requiere trabajo significativo:
                más datos, simplificación del modelo, o reconsiderar el enfoque completo.
                ''' if diff_r2 >= 0.25 or metrics_test['RMSE']/metrics_train['RMSE'] > 2.0 else ''}
                
                {'''
                **MODELO INSUFICIENTE - CAMBIAR ENFOQUE**
                El rendimiento es pobre tanto en Train como en Test (underfitting). El modelo es demasiado simple
                o las variables no son predictivas. Necesitas variables mejores o un modelo más complejo.
                ''' if metrics_test['R2'] < 0.5 and metrics_train['R2'] < 0.6 else ''}

                """)
        
        comparacion_conjuntos = pd.DataFrame({
            'Conjunto': ['Entrenamiento', 'Prueba'],
            'R²': [metrics_train['R2'], metrics_test['R2']],
            'RMSE': [metrics_train['RMSE'], metrics_test['RMSE']],
            'MAE': [metrics_train['MAE'], metrics_test['MAE']]
        })
        
        fig4 = go.Figure()
        colores_metrics = {
            'R²': ('#1f77b4', '#ff7f0e'),
            'RMSE': ('#2ca02c', '#d62728'),
            'MAE': ('#9467bd', '#8c564b')
        }
        
        for idx, metric in enumerate(['R²', 'RMSE', 'MAE']):
            valores = comparacion_conjuntos[metric].values
            
            fig4.add_trace(go.Bar(
                name=f'{metric} - Train', x=['Entrenamiento'], y=[valores[0]],
                text=[f'{valores[0]:.4f}'], textposition='auto',
                marker_color=colores_metrics[metric][0],
                legendgroup=metric, offsetgroup=idx
            ))
            
            fig4.add_trace(go.Bar(
                name=f'{metric} - Test', x=['Prueba'], y=[valores[1]],
                text=[f'{valores[1]:.4f}'], textposition='auto',
                marker_color=colores_metrics[metric][1],
                legendgroup=metric, offsetgroup=idx
            ))
        
        fig4.update_layout(
            title=f"Comparación: Train ({len(y_train_original)} obs) vs Test ({len(y_test_original)} obs)",
            xaxis_title="Conjunto de Datos",
            yaxis_title="Valor de Métrica",
            barmode='group',
            hovermode='x unified',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1)
        )
        
        # Diagnóstico de overfitting
        diagnostico_texto = f"ΔR² = {diff_r2:.4f}"
        if diff_r2 > 0.15:
            diagnostico_texto += " - OVERFITTING"
            color_diag = "red"
        elif diff_r2 > 0.10:
            diagnostico_texto += " - Overfitting leve"
            color_diag = "orange"
        else:
            diagnostico_texto += " - Buena generalización"
            color_diag = "green"
        
        fig4.add_annotation(
            x=0.5, y=0.95, xref='paper', yref='paper',
            text=diagnostico_texto, showarrow=False,
            bgcolor=color_diag, font=dict(color='white', size=14), opacity=0.8
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Resumen final
        st.subheader("Resumen del Análisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset:**")
            st.markdown(f"""
            - Observaciones totales: {n_original}
            - Observaciones finales: {len(df)}
            - Variables predictoras: {len(x_col)}
            - Outliers eliminados: {n_outliers if remove_outliers else 0}
            """)
            
            st.markdown("**Configuración:**")
            st.markdown(f"""
            - Normalización: {normalize_method}
            - Test size: {test_size*100:.0f}%
            - CV Folds: {cv_folds}
            - Random state: {random_state}
            """)
        
        with col2:
            st.markdown("**Rendimiento (Test):**")
            st.markdown(f"""
            - R²: {metrics_test['R2']:.4f}
            - RMSE: {metrics_test['RMSE']:.4f}
            - MAE: {metrics_test['MAE']:.4f}
            - MAPE: {metrics_test['MAPE']:.2f}%
            """)
            
            st.markdown("**Diagnóstico:**")
            overfitting_status = 'Sí' if diff_r2 > 0.15 else 'No'
            rmse_relativo = (metrics_test['RMSE'] / (y_max_original - y_min_original)) * 100
            st.markdown(f"""
            - Overfitting: {overfitting_status}
            - Error relativo: {rmse_relativo:.2f}% del rango
            - Convergencia: {'Exitosa' if n_iteraciones < 300 else 'Límite alcanzado'}
            """)
        
        st.success("Análisis completado exitosamente")
        
    except Exception as e:
        st.error(f"Error durante la ejecución: {str(e)}")
        st.exception(e)
        
        with st.expander("Información de Debug", expanded=False):
            st.write("**Variables en el error:**")
            st.write(f"- Forma del DataFrame: {df.shape if 'df' in locals() else 'N/A'}")
            st.write(f"- Columnas X: {x_col if 'x_col' in locals() else 'N/A'}")
            st.write(f"- Columna Y: {y_col if 'y_col' in locals() else 'N/A'}")
            st.write(f"- Valores nulos: {df.isnull().sum().sum() if 'df' in locals() else 'N/A'}")