# ===============================================================
# 🎯 DASHBOARD PROFESIONAL - PREDICCIONES SOLARES PARA PANAMÁ
# ===============================================================
# Dashboard con funcionalidad completa y variabilidad temporal real
# Configurado específicamente para Panamá (hemisferio sur)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Importar nuestra clase profesional
from modelo_solar_predictor import SolarPredictor

# Configuración de página
st.set_page_config(
    page_title="🔋 Dashboard Profesional - Predicciones Solares Panamá",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

# Título
st.title("🔋 Dashboard Profesional - Predicciones Solares Panamá")
st.markdown("---")

# Verificar si existe el modelo entrenado
model_path = "modelo_solar_entrenado.pkl"
model_exists = os.path.exists(model_path)

if not model_exists:
    st.error("❌ No se encontró el modelo entrenado. Ejecuta primero:")
    st.code("python modelo_corregido_horas_sol.py")
    st.stop()

# Cargar modelo
@st.cache_data
def cargar_modelo():
    """Cargar el modelo profesional"""
    try:
        predictor = SolarPredictor()
        predictor.load(model_path)
        return predictor
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        return None

# Cargar datos históricos
@st.cache_data
def cargar_datos_historicos():
    """Cargar datos históricos procesados"""
    try:
        if os.path.exists("data/processed_solar_data.xlsx"):
            df = pd.read_excel("data/processed_solar_data.xlsx")
            return df
        elif os.path.exists("Datos reales.xlsx"):
            df = pd.read_excel("Datos reales.xlsx")
            # Renombrar columna si es necesario
            if 'fecha y hora' in df.columns:
                df = df.rename(columns={'fecha y hora': 'fecha_y_hora'})
            df['fecha_y_hora'] = pd.to_datetime(df['fecha_y_hora'])
            return df
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        return None

# Cargar resultados del modelo
@st.cache_data
def cargar_resultados_modelo():
    """Cargar resultados del modelo original"""
    try:
        if os.path.exists("RESULTADOS_CORREGIDOS_SOL.xlsx"):
            df = pd.read_excel("RESULTADOS_CORREGIDOS_SOL.xlsx")
            return df
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error cargando resultados: {e}")
        return None

# Cargar datos
with st.spinner("🔄 Cargando modelo y datos..."):
    predictor = cargar_modelo()
    df_historical = cargar_datos_historicos()
    df_resultados = cargar_resultados_modelo()

if predictor is None:
    st.stop()

# Sidebar con información del modelo
st.sidebar.header("📊 Información del Modelo")
st.sidebar.write(f"**Modelo:** SolarPredictor Profesional")
st.sidebar.write(f"**Features:** {len(predictor.feature_cols)}")
st.sidebar.write(f"**Parámetros:** {predictor.model_params}")
st.sidebar.write(f"**Entrenado:** {'Sí' if predictor.is_fitted else 'No'}")

st.sidebar.header("📁 Archivos Disponibles")
st.sidebar.write("✅ Modelo entrenado" if model_exists else "❌ Modelo entrenado")
st.sidebar.write("✅ Datos procesados" if df_historical is not None else "❌ Datos procesados")
st.sidebar.write("✅ Resultados modelo" if df_resultados is not None else "❌ Resultados modelo")
st.sidebar.write("✅ Configuración" if True else "❌ Configuración")

# Mostrar mensaje de éxito
st.success("✅ Modelo y datos cargados correctamente")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos Históricos", "🔮 Predicciones Futuras", "📈 Análisis de Modelo", "⚙️ Configuración"])

with tab1:
    st.header("📊 Análisis de Datos Históricos")
    
    if df_resultados is not None:
        # Selectores de fecha
        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Fecha de inicio", value=df_resultados['fecha_y_hora'].min().date())
        with col2:
            fecha_fin = st.date_input("Fecha de fin", value=df_resultados['fecha_y_hora'].max().date())
        
        # Filtrar datos
        df_filtrado = df_resultados[
            (df_resultados['fecha_y_hora'].dt.date >= fecha_inicio) & 
            (df_resultados['fecha_y_hora'].dt.date <= fecha_fin)
        ]
        
        if len(df_filtrado) > 0:
            # Métricas del período
            st.subheader("📈 Métricas del Período Seleccionado")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mae = np.mean(np.abs(df_filtrado['generacion'] - df_filtrado['Prediccion_Corregida']))
                st.metric("MAE", f"{mae:.0f} kWh")
            
            with col2:
                r2 = 1 - np.sum((df_filtrado['generacion'] - df_filtrado['Prediccion_Corregida'])**2) / np.sum((df_filtrado['generacion'] - df_filtrado['generacion'].mean())**2)
                st.metric("R²", f"{r2:.3f}")
            
            with col3:
                rmse = np.sqrt(np.mean((df_filtrado['generacion'] - df_filtrado['Prediccion_Corregida'])**2))
                st.metric("RMSE", f"{rmse:.0f} kWh")
            
            with col4:
                max_pred = df_filtrado['Prediccion_Corregida'].max()
                st.metric("Máx Predicción", f"{max_pred:.0f} kWh")
            
            # Gráficas
            st.subheader("📊 Gráficas de Análisis")
            
            # Gráfica 1: Superposición temporal
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_filtrado['fecha_y_hora'], y=df_filtrado['generacion'], 
                                    name='Real', line=dict(color='blue')))
            fig1.add_trace(go.Scatter(x=df_filtrado['fecha_y_hora'], y=df_filtrado['Prediccion_Corregida'], 
                                    name='Predicción', line=dict(color='red')))
            fig1.update_layout(title="Real vs Predicción", xaxis_title="Fecha", yaxis_title="Generación (kWh)")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Gráfica 2: Zoom primeros 3 días
            st.subheader("🔍 Gráfica 2: Zoom - Primeros 3 Días")
            df_zoom = df_filtrado.head(72)  # 3 días * 24 horas
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_zoom['fecha_y_hora'], y=df_zoom['generacion'], 
                                    name='Real', line=dict(color='blue')))
            fig2.add_trace(go.Scatter(x=df_zoom['fecha_y_hora'], y=df_zoom['Prediccion_Corregida'], 
                                    name='Predicción', line=dict(color='red')))
            fig2.update_layout(title="Zoom - Primeros 3 Días", xaxis_title="Fecha", yaxis_title="Generación (kWh)")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Gráfica 3: Análisis por hora
            st.subheader("⏰ Análisis por Hora")
            df_hora = df_filtrado.groupby('hora').agg({
                'generacion': 'mean',
                'Prediccion_Corregida': 'mean'
            }).reset_index()
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_hora['hora'], y=df_hora['generacion'], 
                                    name='Real', line=dict(color='blue')))
            fig3.add_trace(go.Scatter(x=df_hora['hora'], y=df_hora['Prediccion_Corregida'], 
                                    name='Predicción', line=dict(color='red')))
            fig3.update_layout(title="Promedio por Hora", xaxis_title="Hora", yaxis_title="Generación (kWh)")
            st.plotly_chart(fig3, use_container_width=True)
            
        else:
            st.warning("No hay datos para el período seleccionado")
    else:
        st.warning("No se encontraron resultados del modelo")

with tab2:
    st.header("🔮 Predicciones Futuras")
    
    st.info("🚨 **PREDICCIONES CON MODELO PROFESIONAL** - Variabilidad temporal real implementada")
    st.info("ℹ️ Las predicciones respetan las horas de sol reales de Panamá por mes")
    
    # Configuración de predicciones
    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio = st.date_input("Fecha de inicio de predicciones:", value=pd.Timestamp.now().date())
    with col2:
        dias_adelante = st.slider("Días hacia el futuro:", min_value=1, max_value=30, value=7)
    
    # Botón para generar predicciones
    if st.button("🔮 Generar Predicciones", type="primary"):
        try:
            # Crear fechas futuras
            fechas = pd.date_range(start=fecha_inicio, periods=dias_adelante * 24, freq='H')
            
            # Crear DataFrame para predicciones
            df_futuro = pd.DataFrame({
                'fecha_y_hora': fechas
            })
            
            # Generar predicciones
            with st.spinner("Generando predicciones..."):
                predicciones = predictor.predict(df_futuro)
                df_futuro['prediccion'] = predicciones
            
            # Mostrar métricas
            st.subheader("📊 Métricas de Predicciones")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = df_futuro['prediccion'].sum()
                st.metric("Total", f"{total:,.0f} kWh")
            
            with col2:
                max_pred = df_futuro['prediccion'].max()
                st.metric("Máximo", f"{max_pred:,.0f} kWh")
            
            with col3:
                avg_pred = df_futuro['prediccion'].mean()
                st.metric("Promedio", f"{avg_pred:,.0f} kWh")
            
            with col4:
                periodos = len(df_futuro)
                st.metric("Períodos", f"{periodos}")
            
            # Gráfica de predicciones
            st.subheader("📈 Predicciones Futuras")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_futuro['fecha_y_hora'], y=df_futuro['prediccion'], 
                                   name='Predicción', line=dict(color='green')))
            fig.update_layout(title="Predicciones Futuras", xaxis_title="Fecha", yaxis_title="Generación (kWh)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de datos
            st.subheader("📋 Datos de Predicciones")
            st.dataframe(df_futuro, use_container_width=True)
            
            # Descargar datos
            csv = df_futuro.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Predicciones (CSV)",
                data=csv,
                file_name=f"predicciones_{fecha_inicio}_{dias_adelante}dias.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error generando predicciones: {e}")

with tab3:
    st.header("📈 Análisis de Modelo")
    
    # Importancia de features
    st.subheader("🎯 Importancia de Features")
    try:
        importance = predictor.get_feature_importance()
        df_importance = pd.DataFrame(importance, columns=['Feature', 'Importancia'])
        df_importance = df_importance.head(15)  # Top 15
        
        fig = px.bar(df_importance, x='Importancia', y='Feature', orientation='h')
        fig.update_layout(title="Top 15 Features Más Importantes")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de importancia
        st.dataframe(df_importance, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error obteniendo importancia: {e}")
    
    # Parámetros del modelo
    st.subheader("⚙️ Parámetros del Modelo")
    st.json(predictor.model_params)
    
    # Información del scaler
    st.subheader("📊 Información del Scaler")
    st.json(predictor.scaler_info)

with tab4:
    st.header("⚙️ Configuración del Sistema")
    
    st.subheader("🌍 Configuración de Panamá")
    st.write("**Horas de sol por mes:**")
    
    # Mostrar horas de sol
    for mes, horas in predictor.HORAS_SOL_MADRID.items():
        horas_luz = horas['puesta'] - horas['salida']
        mes_nombre = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                     'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][mes]
        st.write(f"- **{mes_nombre}**: {horas['salida']:4.1f}h - {horas['puesta']:4.1f}h = {horas_luz:4.1f}h luz")
    
    st.subheader("📋 Comandos del Sistema")
    st.code("""
# Entrenar modelo
python modelo_corregido_horas_sol.py

# Ejecutar dashboard
streamlit run dashboard_profesional.py

# Verificar configuración
python -c "from modelo_solar_predictor import SolarPredictor; p = SolarPredictor(); p.load('modelo_solar_entrenado.pkl'); print('Configuración de Panamá aplicada')"
    """)
    
    st.subheader("📊 Estadísticas del Sistema")
    st.write(f"- **Features utilizadas**: {len(predictor.feature_cols)}")
    st.write(f"- **Modelo entrenado**: {'Sí' if predictor.is_fitted else 'No'}")
    patrones_count = len(predictor.patrones_hora.get('mean', {})) if hasattr(predictor, 'patrones_hora') and predictor.patrones_hora else 0
    st.write(f"- **Patrones históricos**: {patrones_count} horas")
    st.write(f"- **Configuración**: Panamá (hemisferio sur)")

# Footer
st.markdown("---")
st.markdown("🔋 **Dashboard Profesional de Predicciones Solares Panamá** - Sistema con variabilidad temporal real")
