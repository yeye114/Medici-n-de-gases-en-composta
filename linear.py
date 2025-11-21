import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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
    r2 = r2_score(y_true, y_pred)
    
    # Asegurar que R² esté entre 0 y 1
    r2 = max(0.0, min(1.0, r2))
    
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2,
        'MAPE': np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100,
        'Max_Error': np.max(np.abs(y_true - y_pred)),
        'Min_Error': np.min(np.abs(y_true - y_pred)),
        'Std_Error': np.std(y_true - y_pred),
        'Adjusted_R2': max(0.0, min(1.0, 1 - (ss_res / ss_tot)))
    }


# ==================== FUNCIÓN PRINCIPAL ====================

def ejecutar(df: pd.DataFrame, x_col: list, y_col: str, test_size: float = 0.2,
             normalize_method: str = 'standard', modelo_tipo: str = "linear",
             remove_outliers: bool = False, outlier_method: str = 'iqr',
             outlier_threshold: float = 1.5, impute_strategy: str = 'median',
             cv_folds: int = 5, random_state: int = 42):
    """Regresión Lineal con análisis completo y visualizaciones profesionales"""
    
    try:
        # Convertir parámetros a tipos correctos con manejo de errores
        try:
            test_size = float(test_size)
        except (ValueError, TypeError):
            test_size = 0.7  
            
        try:
            outlier_threshold = float(outlier_threshold)
        except (ValueError, TypeError):
            outlier_threshold = 1.5
            
        try:
            cv_folds = int(cv_folds)
        except (ValueError, TypeError):
            cv_folds = 5
            
        try:
            random_state = int(random_state)
        except (ValueError, TypeError):
            random_state = 42
        
        st.title("Análisis de Regresión Lineal")
        
        # ============================================
        # INFORMACIÓN DEL ALGORITMO (EXPANDIDA)
        # ============================================
        with st.expander("Información sobre Regresión Lineal", expanded=False):
            st.markdown("""
            ### ¿Qué es Regresión Lineal?
            
            La **Regresión Lineal** es un algoritmo de aprendizaje supervisado que modela la relación entre 
            una variable dependiente (objetivo) y una o más variables independientes (predictoras) mediante 
            una ecuación lineal.
            
            ---
            
            ### Cómo Funciona
            
            1. **Objetivo:** Minimizar la suma de errores cuadráticos (MSE)
            2. **Método:** Mínimos Cuadrados Ordinarios (OLS)
            3. **Optimización:** Encuentra los coeficientes β que minimizan la función de costo
            4. **Supuestos:**
               - Relación lineal entre variables
               - Independencia de errores
               - Homocedasticidad (varianza constante)
               - Normalidad de residuos
            
            ---
            
            ### Ventajas
            
            - **Interpretabilidad:** Fácil de entender y explicar
            - **Velocidad:** Entrenamiento muy rápido
            - **Eficiencia:** Requiere pocos recursos computacionales
            - **Baseline:** Excelente punto de partida para comparaciones
            - **Estabilidad:** Resultados consistentes y reproducibles
            - **Inferencia:** Permite análisis estadístico de coeficientes
            
            ---
            
            ### Desventajas
            
            - **Linealidad:** Solo captura relaciones lineales
            - **Sensibilidad:** Afectado por outliers extremos
            - **Multicolinealidad:** Problemas con variables correlacionadas
            - **Extrapolación:** No confiable fuera del rango de entrenamiento
            - **Capacidad limitada:** No captura patrones complejos
            """)
        
        # Validación
        with st.spinner("Validando datos..."):
            es_valido, mensaje = validar_datos(df, x_col, y_col)
            if not es_valido:
                st.error(f"Error de validación: {mensaje}")
                return
        
        # ============================================
        # INFORMACIÓN DEL DATASET
        # ============================================
        st.write("---")
        st.header("Información del Dataset")
        
        col_data1, col_data2, col_data3, col_data4 = st.columns(4)
        
        with col_data1:
            st.metric("Total de filas", f"{len(df):,}")
        
        with col_data2:
            st.metric("Variables predictoras", len(x_col))
        
        with col_data3:
            st.metric("Tamaño de prueba", f"{test_size*100:.0f}%")
            st.caption(f"Random state: {random_state}")
        
        with col_data4:
            st.metric("CV Folds", cv_folds)
            st.caption("Validación cruzada")
        
        # ============================================
        # MANEJO DE VALORES NULOS
        # ============================================
        total_nulls = df[x_col + [y_col]].isnull().sum().sum()
        if total_nulls > 0:
            st.write("---")
            st.subheader("Manejo de Valores Nulos")
            null_info = df[x_col + [y_col]].isnull().sum()
            null_info = null_info[null_info > 0]
            st.warning(f"Se encontraron {total_nulls} valores nulos en {len(null_info)} columnas")
            
            col1, col2 = st.columns([2, 1])
            col1.dataframe(null_info.to_frame('Valores Nulos'), use_container_width=True)
            col2.info(f"**Estrategia:** {impute_strategy}")
            
            df = imputar_valores_faltantes(df, x_col, y_col, impute_strategy)
            st.success(f"Valores nulos imputados usando '{impute_strategy}'")
        
        # ============================================
        # ANÁLISIS DE OUTLIERS
        # ============================================
        st.write("---")
        st.header("Análisis de Outliers")
        
        df_original = df.copy()
        n_original = len(df_original)
        
        df_clean, outliers_mask, outliers_info = detectar_outliers(
            df, y_col, method=outlier_method, threshold=outlier_threshold
        )
        n_outliers = outliers_info['n_outliers']
        porcentaje_outliers = (n_outliers / n_original * 100) if n_original > 0 else 0
        
        # Métricas en columnas
        col_out1, col_out2, col_out3, col_out4 = st.columns(4)
        
        with col_out1:
            st.metric("Total de datos", f"{n_original:,}")
        
        with col_out2:
            st.metric("Outliers detectados", f"{n_outliers}")
            st.caption(f"{porcentaje_outliers:.2f}% del total")
        
        with col_out3:
            if 'lower_bound' in outliers_info:
                st.metric("Límite inferior", f"{outliers_info['lower_bound']:.4f}")
        
        with col_out4:
            if 'upper_bound' in outliers_info:
                st.metric("Límite superior", f"{outliers_info['upper_bound']:.4f}")
        
        if n_outliers > 0:
            with st.expander("Ver outliers detectados", expanded=False):
                # Mostrar mensaje de advertencia DENTRO del desplegable
                if porcentaje_outliers > 5:
                    st.warning(f"Se detectaron {n_outliers} outliers ({porcentaje_outliers:.2f}%). "
                              "Considera revisar o transformar los datos.")
                else:
                    st.info(f"Nivel de outliers aceptable ({porcentaje_outliers:.2f}%).")
                
                st.dataframe(outliers_info['outliers_df'], use_container_width=True)
            
            if remove_outliers:
                df = df[~outliers_mask]
                st.success(f"Se eliminaron {n_outliers} outliers. Datos restantes: {len(df)}")
        else:
            st.success("No se detectaron outliers en los datos.")
        
        # Verificar datos suficientes
        min_samples = max(10, len(x_col) * 5)
        if len(df) < min_samples:
            st.error(f"Datos insuficientes ({len(df)} observaciones). Se recomiendan al menos {min_samples}.")
            return
        
        # ============================================
        # PREPARACIÓN DE DATOS
        # ============================================
        X = df[x_col].values
        y = df[y_col].values
        y_min_original, y_max_original = y.min(), y.max()
        y_range = y_max_original - y_min_original
        
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
        
        # ============================================
        # ENTRENAMIENTO DEL MODELO
        # ============================================
        with st.spinner(f"Entrenando modelo {modelo_tipo.capitalize()}..."):
            if modelo_tipo == "linear":
                model = LinearRegression()
            elif modelo_tipo == "ridge":
                model = Ridge(alpha=1.0, random_state=random_state)
            elif modelo_tipo == "lasso":
                model = Lasso(alpha=0.1, random_state=random_state, max_iter=10000)
            elif modelo_tipo == "elastic":
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=10000)
            else:
                st.error("Modelo no reconocido. Usa: 'linear', 'ridge', 'lasso' o 'elastic'.")
                return
            
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
        
        # ============================================
        # MÉTRICAS DE RENDIMIENTO (CON EXPLICACIONES)
        # ============================================
        st.write("---")
        st.header("Métricas de Rendimiento")
        
        # Explicación de métricas
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
        
        metrics_test = calcular_metricas(y_test_original, y_pred_test_original)
        metrics_train = calcular_metricas(y_train_original, y_pred_train_original)
        
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
                    scoring='r2', n_jobs=-1
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                st.warning(f"No se pudo completar la validación cruzada: {str(e)}")
                cv_scores = None
                cv_mean = None
                cv_std = None
        
        # ============================================
        # GRÁFICO 1: VALORES REALES VS PREDICHOS
        # ============================================
        st.write("---")
        st.subheader("Visualizaciones de Predicciones")
        st.markdown("#### Valores Reales vs Predichos")
        
        # Explicación del gráfico 1
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown("""
            ### Valores Reales vs Predichos
            
            **¿Qué muestra este gráfico?**
            
            Este gráfico compara las predicciones del modelo contra los valores reales en el conjunto de prueba.
            
            ---
            
            **Ejes del gráfico:**
            
            - **Eje X (horizontal):** Valores reales de la variable objetivo (lo que realmente sucedió)
            - **Eje Y (vertical):** Valores predichos por el modelo (lo que el modelo estimó)
            - **Numeración:** Los valores representan las unidades originales de tu variable objetivo
            
            ---
            
            **Elementos visuales:**
            
            - **Puntos (círculos):** Cada punto representa una observación del conjunto de prueba
            - **Color de los puntos:** Indica el error absoluto (rojo más intenso = mayor error)
            - **Tamaño de los puntos:** Proporcional al error absoluto (puntos más grandes = errores más grandes)
            - **Línea verde diagonal (y=x):** Representa la predicción perfecta
            
            ---
            
            **¿Cómo interpretar?**
            
            **Modelo ideal:**
            - Todos los puntos estarían exactamente sobre la línea verde
            - Esto significaría predicciones perfectas (Real = Predicho)
            
            **Buen modelo:**
            - Los puntos se concentran cerca de la línea verde
            - Poca dispersión alrededor de la línea
            - Colores claros (errores pequeños)
            
            **Modelo con problemas:**
            - Puntos muy alejados de la línea verde
            - Gran dispersión
            - Muchos puntos de color rojo intenso
            - Patrones sistemáticos (por ejemplo, todos los puntos debajo o encima de la línea)
            
            ---
            **La línea verde (y = x):**
            
            - **Ecuación:** y = x (pendiente = 1, intercepto = 0)
            - **Significado:** Si el modelo predice exactamente el valor real, el punto caerá sobre esta línea
            - **Ejemplo:** Si el valor real es 100, y el modelo predice 100, el punto está en (100, 100) sobre la línea
            - **Interpretación:** Cuanto más cerca estén los puntos de esta línea, mejores son las predicciones
            
            ---
            
            **¿Por qué R² siempre está entre 0 y 1?**
            
            En esta implementación, R² está limitado al rango [0, 1]:
            
            - **R² = 1.0:** Predicción perfecta (todos los puntos en la línea verde)
            - **R² = 0.8-0.99:** Excelente modelo
            - **R² = 0.5-0.8:** Buen modelo
            - **R² = 0.0-0.5:** Modelo débil
            - **R² = 0.0:** El modelo no es mejor que predecir la media
            
            **Fórmula:** R² = 1 - (SS_residual / SS_total)
            
            Donde:
            - SS_residual = suma de errores cuadráticos del modelo
            - SS_total = varianza total de los datos
            
            **Si obtienes R² = 0.0:**
            - El modelo predice igual que la media
            - Considera cambiar el modelo o revisar las variables
            - Puede indicar que no hay relación lineal entre X e y

            ---
           

            **Patrones a identificar:**
            
            1. **Subestimación sistemática:** Puntos principalmente debajo de la línea → el modelo predice valores menores
            2. **Sobreestimación sistemática:** Puntos principalmente encima de la línea → el modelo predice valores mayores
            3. **Heterocedasticidad:** Dispersión que aumenta o disminuye a lo largo del rango → variabilidad no constante
            4. **No linealidad:** Forma curva en la distribución de puntos → relación no lineal entre variables
            
            ---
            
            **Métricas mostradas:**
            
            - **R²:** Qué tan bien los puntos se ajustan a la línea (cercano a 1 = mejor)
            - **Error Promedio:** Distancia promedio de los puntos a la línea verde
            """)
        
        comparacion_df = pd.DataFrame({
            "Valor Real": y_test_original,
            "Valor Predicho": y_pred_test_original,
            "Error": y_test_original - y_pred_test_original,
            "Error Absoluto": np.abs(y_test_original - y_pred_test_original)
        })
        
        n_puntos = len(y_test_original)
        error_promedio = comparacion_df["Error Absoluto"].mean()
        
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
        
        min_val = min(y_test_original.min(), y_pred_test_original.min())
        max_val = max(y_test_original.max(), y_pred_test_original.max())
        
        fig1.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Predicción Perfecta (y=x)',
            line=dict(color='green', dash='dash', width=2.5)
        ))
        
         # Formatear R² mostrando siempre el porcentaje
        r2_value = metrics_test["R2"]
        r2_percent = r2_value * 100
        
        fig1.update_layout(annotations=[
            dict(
                x=0.02, y=0.98, xref='paper', yref='paper',
                text=f'R² = {r2_percent:.2f}%<br>Error Promedio = {error_promedio:.4f}',
                showarrow=False, bgcolor='rgba(142, 174, 191, 0.8)',
                borderwidth=1, borderpad=15
            ),
        ])
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # ============================================
        # ECUACIONES MATEMÁTICAS (DENTRO DE EXPANDER)
        # ============================================
        with st.expander("Ecuaciones Matemáticas Utilizadas", expanded=False):
            # Crear dos columnas para las ecuaciones
            col_ecuacion1, col_ecuacion2 = st.columns(2)
            
            # ========== COLUMNA 1: ECUACIÓN DEL MODELO ==========
            with col_ecuacion1:
                st.markdown("#### Ecuación del Modelo Entrenado")
                
                # Obtener coeficientes e intercepto
                coeficientes = model.coef_
                intercepto = model.intercept_
                
                # Construir ecuación en formato LaTeX
                if len(x_col) == 1:
                    latex_eq = f"\\hat{{y}} = {intercepto:.4f} + {coeficientes[0]:.4f} \\cdot x"
                else:
                    latex_eq = f"\\hat{{y}} = {intercepto:.4f}"
                    for i, coef in enumerate(coeficientes, 1):
                        signo = "+" if coef >= 0 else ""
                        latex_eq += f" {signo} {coef:.4f} \\cdot x_{i}"
                
                # Mostrar ecuación
                st.latex(latex_eq)
                
                st.markdown("""
                **Descripción:**
                - Esta es la ecuación que el modelo aprendió de los datos
                - Representa la relación encontrada entre variables predictoras y objetivo
                - Los coeficientes indican el peso/importancia de cada variable
                """)
                
                # Detalles de coeficientes en expander interno
                with st.expander("Ver coeficientes detallados"):
                    coef_df = pd.DataFrame({
                        'Variable': ['Intercepto (β₀)'] + [f'x_{i+1} ({var})' for i, var in enumerate(x_col)],
                        'Coeficiente': [intercepto] + list(coeficientes),
                        'Magnitud': [abs(intercepto)] + [abs(c) for c in coeficientes],
                        'Efecto': ['Constante'] + ['Positivo ↑' if c > 0 else 'Negativo ↓' for c in coeficientes]
                    })
                    
                    # Ordenar por magnitud
                    coef_df_sorted = coef_df.iloc[1:].sort_values('Magnitud', ascending=False)
                    coef_df_final = pd.concat([coef_df.iloc[[0]], coef_df_sorted])
                    
                    st.dataframe(coef_df_final.style.format({
                        'Coeficiente': '{:.6f}',
                        'Magnitud': '{:.6f}'
                    }).background_gradient(subset=['Magnitud'], cmap='YlOrRd'), 
                    use_container_width=True)
                    
                    # Variable más influyente
                    if len(coeficientes) > 0:
                        idx_max = np.argmax([abs(c) for c in coeficientes])
                        var_max = x_col[idx_max]
                        coef_max = coeficientes[idx_max]
                        st.success(f"**Variable más influyente:** {var_max} (coef: {coef_max:.6f})")
            
            # ========== COLUMNA 2: LÍNEA DE REFERENCIA y=x ==========
            with col_ecuacion2:
                st.markdown("#### Línea de Predicción Perfecta")
                
                # Ecuación de la línea perfecta
                st.latex(r"y = x")
                
                st.markdown("""
                **Descripción:**
                - Esta NO es la ecuación del modelo, es una línea de referencia
                - Representa el escenario ideal donde el modelo predice perfectamente
                - Significado: Valor Predicho = Valor Real
                
                **Interpretación:**
                - **Sobre la línea:** Predicción perfecta (error = 0)
                - **Cerca de la línea:** Buenas predicciones
                - **Lejos de la línea:** Predicciones con errores grandes
                
                **Propiedades matemáticas:**
                """)
                
                # Tabla de propiedades
                props_df = pd.DataFrame({
                    'Propiedad': ['Pendiente (m)', 'Intercepto (b)', 'Ángulo', 'Dominio', 'Rango'],
                    'Valor': ['1', '0', '45°', '(-∞, +∞)', '(-∞, +∞)']
                })
                
                st.dataframe(props_df, use_container_width=True, hide_index=True)
                
                st.info("**Nota importante:** Cuanto más cerca estén los puntos del gráfico de esta línea verde, mejor es el modelo.")

        # ============================================
        # GRÁFICO 2: RESIDUOS VS PREDICHOS
        # ============================================
        st.markdown("#### Residuos vs Valores Predichos")
        
        # Explicación del gráfico 2
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown("""
            ### Análisis de Residuos
            
            **¿Qué muestra este gráfico?**
            
            Este gráfico muestra los errores (residuos) del modelo a través del rango de predicciones.
            
            ---
            
            **Ejes del gráfico:**
            
            - **Eje X (horizontal):** Valores predichos por el modelo
            - **Eje Y (vertical):** Residuos = (Valor Real - Valor Predicho)
            - **Numeración del eje X:** Unidades de la variable objetivo predicha
            - **Numeración del eje Y:** Unidades de error (misma escala que la variable objetivo)
            
            ---
            
            **Elementos visuales:**
            
            - **Puntos azules:** Cada punto es un residuo (error) de una observación
            - **Línea roja horizontal (y=0):** Representa error cero (predicción perfecta)
            - **Banda verde (±1 std):** Contiene aproximadamente el 68% de los errores
            - **Banda amarilla (±2 std):** Contiene aproximadamente el 95% de los errores
            
            ---
            
            **¿Cómo interpretar?**
            
            **Modelo ideal (cumple supuestos):**
            - Puntos distribuidos aleatoriamente alrededor de y=0
            - Sin patrones visibles (parece "nube de puntos")
            - Variabilidad constante en todo el rango (homocedasticidad)
            - La mayoría de puntos dentro de las bandas coloreadas
            
            **Problemas a detectar:**
            
            1. **Patrón en forma de embudo:**
               - Dispersión aumenta o disminuye → heterocedasticidad
               - Solución: transformar datos o usar modelos robustos
            
            2. **Patrón curvo:**
               - Forma de U o parábola → relación no lineal
               - Solución: agregar términos cuadráticos o usar modelos no lineales
            
            3. **Puntos principalmente arriba o abajo de cero:**
               - Sesgo sistemático en las predicciones
               - Modelo consistentemente sobre/subestima
            
            4. **Puntos muy alejados de las bandas:**
               - Outliers o valores atípicos
               - Pueden requerir investigación o tratamiento especial
            
            ---
            
            **Interpretación de las bandas:**
            
            - **Dentro de la banda verde (±1σ):** Errores normales esperados (~68%)
            - **Dentro de la banda amarilla (±2σ):** Errores aceptables (~95%)
            - **Fuera de la banda amarilla:** Posibles outliers o casos extremos
            
            ---
            
            **¿Qué significan los residuos?**
            
            - **Residuo positivo (+):** El modelo subestimó (predicción < real)
            - **Residuo negativo (-):** El modelo sobreestimó (predicción > real)
            - **Residuo cercano a 0:** Predicción muy precisa
            
            ---
            
            **Métricas mostradas:**
            
            - **Media:** Promedio de los residuos (ideal ≈ 0)
            - **Std:** Desviación estándar de los residuos (menor = mejor consistencia)
            """)
        
        residuos = comparacion_df["Error"]
        residuo_medio = residuos.mean()
        residuo_std = residuos.std()
        
        fig2 = px.scatter(
            x=y_pred_test_original, y=comparacion_df["Error"],
            title=f"Residuos vs Valores Predichos ({n_puntos} observaciones)",
            labels={"x": f"{y_col} (Predicho)", "y": "Residuo (Real - Predicho)"}
        )
        
        fig2.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        
        fig2.add_hrect(
            y0=residuo_medio - residuo_std, 
            y1=residuo_medio + residuo_std,
            fillcolor="green", opacity=0.1, line_width=0
        )
        
        fig2.add_hrect(
            y0=residuo_medio - 2*residuo_std, 
            y1=residuo_medio + 2*residuo_std,
            fillcolor="yellow", opacity=0.05, line_width=0
        )
        
        fig2.update_layout(
            annotations=[dict(
                x=0.01, y=0.98, xref='paper', yref='paper',
                text=f"Media = {residuo_medio:.4f}<br>Std = {residuo_std:.4f}",
                showarrow=False, bgcolor='rgba(142, 174, 191, 0.8)',
                borderwidth=1, borderpad=12, align="left"
            )]
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # ============================================
        # GRÁFICO 3: CONVERGENCIA DEL MODELO
        # ============================================
        st.subheader("Análisis de Convergencia del Modelo")
        
        # Explicación del gráfico 3
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown("""
            ### Curva de Aprendizaje
            
            **¿Qué muestra este gráfico?**
            
            Este gráfico muestra cómo mejora el rendimiento del modelo conforme aumenta la cantidad de datos de entrenamiento.
            
            ---
            
            **Ejes del gráfico:**
            
            - **Eje X (horizontal):** Tamaño del conjunto de entrenamiento (número de observaciones)
            - **Eje Y (vertical):** Score R² (rendimiento del modelo, 0 a 1)
            - **Numeración del eje X:** Cantidad de muestras usadas para entrenar
            - **Numeración del eje Y:** Valor de R² (1.0 = perfecto, 0 = malo)
            
            ---
            
            **Elementos visuales:**
            
            - **Línea azul:** Rendimiento en el conjunto de entrenamiento
            - **Banda azul translúcida:** Variabilidad del score de entrenamiento (±1 desviación estándar)
            - **Línea naranja:** Rendimiento en el conjunto de validación
            - **Banda naranja translúcida:** Variabilidad del score de validación
            
            ---
            
            **¿Cómo interpretar?**
            
            **Modelo ideal:**
            - Ambas líneas convergen hacia un valor alto de R² (cerca de 1)
            - Las líneas se estabilizan (se vuelven horizontales)
            - Gap pequeño entre entrenamiento y validación
            - Bandas de variabilidad estrechas
            
            **Escenarios comunes:**
            
            1. **Alto sesgo (underfitting):**
               - Ambas líneas bajas y estables desde el inicio
               - Convergencia a un R² bajo
               - Poca diferencia entre train y validación
               - **Solución:** Modelo más complejo, más características
            
            2. **Alta varianza (overfitting):**
               - Línea azul (train) mucho más alta que naranja (validación)
               - Gran gap entre las líneas
               - Línea de validación se mantiene baja
               - **Solución:** Más datos, regularización, menos complejidad
            
            3. **Buen ajuste:**
               - Líneas convergen cerca del final
               - Gap mínimo entre train y validación
               - R² alto en validación
               - **Resultado:** Modelo generaliza bien
            
            4. **Necesita más datos:**
               - Líneas aún no convergen al final
               - Línea de validación sigue subiendo
               - **Solución:** Recolectar más datos mejorará el modelo
            
            5. **Datos suficientes:**
               - Líneas se aplanan/estabilizan
               - Agregar más datos no mejorará significativamente
               - **Resultado:** Has alcanzado el límite del modelo
            
            ---
            
            **Análisis de convergencia:**
            
            - **Convergencia exitosa:** Las líneas se juntan y estabilizan
            - **No convergencia:** Las líneas siguen separándose o bajando
            - **Convergencia parcial:** Se acercan pero no se estabilizan
            
            ---
            
            **Bandas de variabilidad (sombreado):**
            
            - **Bandas estrechas:** Modelo consistente, resultados reproducibles
            - **Bandas amplias:** Alta variabilidad, resultados inestables
            - Menor variabilidad en validación = mejor generalización
            
            ---
            
            **Decisiones basadas en este gráfico:**
            
            | Observación | Acción Recomendada |
            |-------------|-------------------|
            | Gap grande entre líneas | Reducir overfitting (regularización, más datos) |
            | Ambas líneas bajas | Aumentar complejidad del modelo |
            | Líneas siguen subiendo | Recolectar más datos |
            | Líneas estables y juntas | Modelo óptimo encontrado |
            | Alta variabilidad (bandas anchas) | Aumentar tamaño del dataset, estabilizar datos |
            """)
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y_scaled,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv_folds, scoring='r2', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=train_sizes, y=train_mean,
            mode='lines+markers', name='Score Entrenamiento',
            line=dict(color='steelblue', width=2.5),
            marker=dict(size=8)
        ))
        
        fig3.add_trace(go.Scatter(
            x=train_sizes, y=train_mean + train_std,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        
        fig3.add_trace(go.Scatter(
            x=train_sizes, y=train_mean - train_std,
            mode='lines', line=dict(width=0),
            fillcolor='rgba(70, 130, 180, 0.2)',
            fill='tonexty', showlegend=False, hoverinfo='skip'
        ))
        
        fig3.add_trace(go.Scatter(
            x=train_sizes, y=val_mean,
            mode='lines+markers', name='Score Validación',
            line=dict(color='darkorange', width=2.5),
            marker=dict(size=8)
        ))
        
        fig3.add_trace(go.Scatter(
            x=train_sizes, y=val_mean + val_std,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        
        fig3.add_trace(go.Scatter(
            x=train_sizes, y=val_mean - val_std,
            mode='lines', line=dict(width=0),
            fillcolor='rgba(255, 140, 0, 0.2)',
            fill='tonexty', showlegend=False, hoverinfo='skip'
        ))
        
        fig3.update_layout(
            title='Curva de Aprendizaje del Modelo',
            xaxis_title='Tamaño del Conjunto de Entrenamiento',
            yaxis_title='Score R²',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # ============================================
        # GRÁFICO 4: COMPARACIÓN TRAIN VS TEST
        # ============================================
        st.subheader("Comparación Train vs Test")
        
        # Explicación del gráfico 4
        with st.expander("¿Cómo interpretar este gráfico?", expanded=False):
            st.markdown("""
            ### Comparación de Rendimiento: Train vs Test
            
            **¿Qué muestra este gráfico?**
            
            Este gráfico compara las métricas de rendimiento del modelo entre el conjunto de entrenamiento y el conjunto de prueba.
            
            ---
            
            **Ejes del gráfico:**
            
            - **Eje X (horizontal):** Tipo de conjunto de datos (Entrenamiento o Prueba)
            - **Eje Y (vertical):** Valor de las métricas de rendimiento
            - **Numeración del eje Y:** Valor numérico de cada métrica (escalas diferentes)
            
            ---
            
            **Elementos visuales:**
            
            - **Barras de colores:** Cada par de barras representa una métrica
            - **Colores por métrica:**
              - R² (azul): Proporción de varianza explicada
              - RMSE (verde): Error cuadrático medio
              - MAE (morado): Error absoluto medio
            - **Números en las barras:** Valor exacto de cada métrica
            - **Etiqueta superior:** Diagnóstico de overfitting con código de color
            
            ---
            
            **¿Cómo interpretar?**
            
            **Modelo ideal:**
            - Barras de Train y Test tienen altura similar en cada métrica
            - R² alto en ambos conjuntos (cercano a 1)
            - RMSE y MAE bajos en ambos conjuntos
            - Diferencias mínimas entre Train y Test
            
            **Análisis por métrica:**
            
            **1. R² (Azul) - MAYOR ES MEJOR:**
            - **Train > Test (pequeña diferencia):** Normal y aceptable
            - **Train >> Test (gran diferencia):** Overfitting - modelo memoriza
            - **Ambos altos y similares:** Excelente generalización
            - **Ambos bajos y similares:** Underfitting - modelo muy simple
            
            **2. RMSE (Verde) - MENOR ES MEJOR:**
            - **Test > Train (pequeña diferencia):** Normal
            - **Test >> Train (gran diferencia):** Overfitting
            - **Ambos bajos:** Predicciones precisas
            - **Ambos altos:** Modelo no captura la relación
            
            **3. MAE (Morado) - MENOR ES MEJOR:**
            - Similar interpretación que RMSE
            - Menos sensible a outliers que RMSE
            - Representa error promedio en unidades originales
            
            ---
            
            **Diagnóstico de Overfitting (etiqueta superior):**
            
            **"Buena generalización" (verde):**
            - ΔR² < 0.10 (diferencia Train-Test menor a 0.10)
            - El modelo funciona bien con datos nuevos
            - **Acción:** Ninguna, el modelo está listo
            
            **"Overfitting leve" (naranja):**
            - 0.10 ≤ ΔR² ≤ 0.15
            - Ligera memorización de datos de entrenamiento
            - **Acción:** Considerar regularización o más datos
            
            **"OVERFITTING" (rojo):**
            - ΔR² > 0.15
            - Modelo memoriza en lugar de generalizar
            - **Acción urgente:**
              - Agregar regularización (Ridge, Lasso)
              - Recolectar más datos
              - Reducir complejidad del modelo
              - Aplicar validación cruzada
              - Eliminar características irrelevantes
            
            ---
            
            **Patrones y sus significados:**
            
            | Patrón Observado | Significado | Acción |
            |------------------|-------------|--------|
            | Train alto, Test bajo | Overfitting | Regularizar/más datos |
            | Ambos bajos | Underfitting | Modelo más complejo |
            | Ambos altos y similares | Perfecto | Modelo listo |
            | Test > Train (en error) | Normal | Aceptable si diferencia pequeña |
            | RMSE >> MAE | Outliers presentes | Revisar datos atípicos |
            
            ---
            
            **Información del título:**
            
            - **Train (X obs):** Número de observaciones usadas para entrenar
            - **Test (Y obs):** Número de observaciones usadas para evaluar
            - Proporción típica: 80/20 o 70/30
            
            ---
            
            **¿Qué buscar?**
            
            ✓ Barras de altura similar entre Train y Test
            ✓ R² alto (>0.7 en muchos casos)
            ✓ RMSE y MAE bajos relativos a los datos
            ✓ Etiqueta verde de "Buena generalización"
            
            ✗ Gran diferencia entre barras Train y Test
            ✗ R² bajo en Test
            ✗ RMSE/MAE muy altos en Test
            ✗ Etiqueta roja de "OVERFITTING"
            """)
        
        diff_r2 = metrics_train['R2'] - metrics_test['R2']
        diff_rmse = metrics_test['RMSE'] - metrics_train['RMSE']
        diff_mae = metrics_test['MAE'] - metrics_train['MAE']
        
        comparacion_conjuntos = pd.DataFrame({
            'Conjunto': ['Entrenamiento', 'Prueba'],
            'R²': [metrics_train['R2'], metrics_test['R2']],
            'RMSE': [metrics_train['RMSE'], metrics_test['RMSE']],
            'MAE': [metrics_train['MAE'], metrics_test['MAE']]
        })
        
        fig4 = go.Figure()
        colores_metrics = {
            'R²': ('#1f77b4', '#1f77b4'),
            'RMSE': ('#2ca02c', '#2ca02c'),
            'MAE': ('#9467bd', '#9467bd')
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
            hovermode='x unified'
        )
        
        # Diagnóstico mejorado que considera tanto R² como overfitting
        diagnostico_texto = f"ΔR² = {diff_r2:.4f}"
        
        # Primero verificar la calidad del modelo (R² de test)
        if metrics_test['R2'] < 0.10:
            # R² muy bajo = modelo muy pobre
            diagnostico_texto += " - MODELO MUY POBRE (R² < 10%)"
            color_diag = "darkred"
            overfitting_status = "No (modelo inútil)"
        elif metrics_test['R2'] < 0.30:
            # R² bajo = modelo pobre
            diagnostico_texto += " - MODELO POBRE (R² < 30%)"
            color_diag = "red"
            overfitting_status = "No (modelo débil)"
        elif diff_r2 > 0.15:
            # Overfitting severo
            diagnostico_texto += " - OVERFITTING SEVERO"
            color_diag = "red"
            overfitting_status = "Sí (severo)"
        elif diff_r2 > 0.10:
            # Overfitting leve
            diagnostico_texto += " - Overfitting leve"
            color_diag = "orange"
            overfitting_status = "Leve"
        elif metrics_test['R2'] < 0.50:
            # Modelo aceptable pero mejorable
            diagnostico_texto += " - Modelo aceptable (R² < 50%)"
            color_diag = "orange"
            overfitting_status = "No (mejorable)"
        elif metrics_test['R2'] < 0.70:
            # Buen modelo
            diagnostico_texto += " - Buen modelo"
            color_diag = "yellowgreen"
            overfitting_status = "No"
        else:
            # Excelente modelo
            diagnostico_texto += " - Excelente modelo"
            color_diag = "green"
            overfitting_status = "No"

        fig4.add_annotation(
            x=0.5, y=0.95, xref='paper', yref='paper',
            text=diagnostico_texto, showarrow=False,
            bgcolor=color_diag, font=dict(color='white', size=14),
            opacity=0.8, borderpad=8
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # ============================================
        # RESUMEN DEL ANÁLISIS (CONSOLIDADO EN DOS COLUMNAS)
        # ============================================
        st.write("---")
        st.header("Resumen del Análisis")
        
        # Calcular error relativo
        error_relativo = (metrics_test['RMSE'] / y_range * 100) if y_range > 0 else 0
        
        # Estado de convergencia
        convergencia = "Exitosa" if metrics_test['R2'] > 0 else "Fallida"
        
        # Crear dos columnas
        col_resumen1, col_resumen2 = st.columns(2)
        
        with col_resumen1:
            st.markdown(f"""
### Dataset y Configuración

**Dataset:**
* Observaciones totales: {n_original}
* Observaciones finales: {len(df)}
* Variables predictoras: {len(x_col)}
* Outliers eliminados: {n_outliers if remove_outliers else 0}

**Configuración:**
* Tipo de modelo: {modelo_tipo.capitalize()}
* Normalización: {normalize_method}
* Test size: {test_size*100:.0f}%
* CV Folds: {cv_folds}
* Random state: {random_state}

**Validación Cruzada:**
* Media R²: {cv_mean:.4f} ± {cv_std:.4f}
* Estabilidad: {'Alta' if cv_std and cv_std < 0.1 else 'Media' if cv_std and cv_std < 0.2 else 'Baja'}
            """)
        
        with col_resumen2:
            st.markdown(f"""
### Rendimiento y Diagnóstico

**Rendimiento (Test):**
* R²: {metrics_test['R2']:.4f}
* RMSE: {metrics_test['RMSE']:.4f}
* MAE: {metrics_test['MAE']:.4f}
* MAPE: {metrics_test['MAPE']:.2f}%

**Rendimiento (Train):**
* R²: {metrics_train['R2']:.4f}
* RMSE: {metrics_train['RMSE']:.4f}
* MAE: {metrics_train['MAE']:.4f}

**Diagnóstico:**
* Overfitting: {overfitting_status}
* Diferencia R² (Train-Test): {diff_r2:.4f}
* Error relativo: {error_relativo:.2f}% del rango
* Convergencia: {convergencia}
            """)
        
        st.success("Análisis completado exitosamente")
        
        # Retornar componentes para hacer predicciones
        return {
            'modelo': model,
            'scaler_X': scaler_X if normalize_method != 'none' else None,
            'scaler_y': scaler_y if normalize_method != 'none' else None,
            'x_col': x_col,
            'metricas': {
                'r2_test': metrics_test['R2'],
                'r2_train': metrics_train['R2'],
                'mae_test': metrics_test['MAE'],
                'rmse_test': metrics_test['RMSE'],
                'mape_test': metrics_test['MAPE'],
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'error_relativo': error_relativo,
                'overfitting': overfitting_status
            }
        }
        
    except Exception as e:
        st.error(f"Error durante la ejecución: {str(e)}")
        st.exception(e)
        
        with st.expander("Información de Debug", expanded=False):
            st.write("**Variables en el error:**")
            st.write(f"- Forma del DataFrame: {df.shape if 'df' in locals() else 'N/A'}")
            st.write(f"- Columnas X: {x_col if 'x_col' in locals() else 'N/A'}")
            st.write(f"- Columna Y: {y_col if 'y_col' in locals() else 'N/A'}")
            st.write(f"- Valores nulos: {df.isnull().sum().sum() if 'df' in locals() else 'N/A'}")


# ==================== FUNCIÓN DE PREDICCIÓN ====================

def predecir_nuevos_datos(modelo_info: dict, nuevos_datos: pd.DataFrame):
    """
    Realiza predicciones con nuevos datos usando el modelo entrenado.
    
    Args:
        modelo_info: Diccionario retornado por ejecutar()
        nuevos_datos: DataFrame con las mismas columnas predictoras
    
    Returns:
        Array con las predicciones
    """
    try:
        modelo = modelo_info['modelo']
        scaler_X = modelo_info['scaler_X']
        scaler_y = modelo_info['scaler_y']
        x_col = modelo_info['x_col']
        
        # Extraer características
        X_nuevo = nuevos_datos[x_col].values
        
        # Escalar si es necesario
        if scaler_X is not None:
            X_nuevo_scaled = scaler_X.transform(X_nuevo)
        else:
            X_nuevo_scaled = X_nuevo
        
        # Predecir
        y_pred_scaled = modelo.predict(X_nuevo_scaled)
        
        # Desescalar
        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_scaled
        
        return y_pred
        
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        return None