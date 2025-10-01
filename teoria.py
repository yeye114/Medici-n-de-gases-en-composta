import streamlit as st

def mostrar():
    # Encabezado general
    st.markdown("<h1>Información Teórica</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #BDC3C7;'>", unsafe_allow_html=True)

    # Nombre del Proyecto
    st.subheader("Nombre del Proyecto")
    st.info("""
    **Prototipo de medición de gases en composta bovino-ovino para estudio comparativo aplicando algoritmo predictivo.**
    """)
    st.markdown("<hr style='border:1px solid #BDC3C7;'>", unsafe_allow_html=True)

    # Delimitación
    st.subheader("Delimitación")
    st.write("""
        Los estudios comparativos permiten establecer relaciones entre dos o más fenómenos o conjuntos de elementos para obtener razones válidas en la explicación de
        diferencias o semejanzas. En nuestro contexto se utilizará para evaluar la viabilidad de uno de los dos tipos de composta que se utilizaran, ya que es del 
        interés de los investigadores conocer cual aporta mayor cantidad de energía bajo condiciones de humedad y temperatura similares
        """)
    st.markdown("<hr style='border:1px solid #BDC3C7;'>", unsafe_allow_html=True)

    # Objetivos
    st.subheader("Objetivos")
    with st.expander("**General:**"):
        st.write("""
        Construir un prototipo para medir mediante sensores variables de temperatura, humedad y gas producidos en compostas bovino y ovino para realizar un estudio 
        comparativo utilizando un algoritmo predictivo.
        """)

    with st.expander("**Específicos:**"):
        st.write("""
        - Implementar un sistema de recolección de datos que registre de manera continua las mediciones de temperatura, humedad y gases, con los cuales realizar un 
        estudio comparativo que permita identificar características de rendimiento entre las compostas bovinas y ovinas.
        """)
    st.markdown("<hr style='border:1px solid #BDC3C7;'>", unsafe_allow_html=True)

    # Justificación
    st.subheader("Justificación")
    st.write("""
        Considerando el eje estratégico de “soberanía alimentaria” establecida por el CONAHCyT dentro de los Programas Nacionales Estratégicos el cual, en su proyecto de 
        incidencia 7 indica “la transición hacia la eliminación del uso de agroquímicos nocivos en actividades agropecuarias y al fortalecimiento de alternativas 
        a la siembra”, Este eje incide en el sistema agroalimentario en su complejidad estructural y dinámica, al considerar que sus determinaciones múltiples y 
        heterogéneas se concretan en Proyectos nacionales de investigación e incidencia (Pronaii) que se desarrollan en territorios específicos, con experiencias 
        locales y regionales de relevancia nacional que, además de plantear objetivos y metas de investigación, proponen metas y acciones de incidencia 
        específicas. (PRONACES 2021). La propuesta de solución es utilizar sensores que midan las variables de temperatura, humedad y gas metano en ambos tipos 
        de compostas, a partir de las cuales realizar un estudio comparativo para determinar la viabilidad de alguna de estas y finalmente establecer una 
        proyección de la producción de gas y su posible utilización.
        """)
    st.markdown("<hr style='border:1px solid #BDC3C7;'>", unsafe_allow_html=True)

    # Lugar e Institución
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lugar del Proyecto")
        st.write("""
        El proyecto se desarrollará en las instalaciones del Instituto Tecnológico Superior del Occidente del Estado de Hidalgo (ITSOEH); ubicado en Paseo del Agrarismo 
        #2000, Carretera Mixquiahuala – Tula km. 2.5 en el municipio de Mixquiahuala de Juárez en el Estado de Hidalgo.
        """)

    with col2:
        st.subheader("Información de la Institución")
        st.write("""
        Organismo Público Descentralizado del Sistema de Educación Superior en el Estado de Hidalgo que atiende el modelo educativo del Subsistema de Educación de 
        Institutos Tecnológicos Descentralizados.
        """)
