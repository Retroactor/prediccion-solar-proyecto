# ===============================================================
# 🎯 DASHBOARD INTERACTIVO - EXPLORACIÓN DE PREDICCIONES
# ===============================================================
# Dashboard para explorar predicciones de manera interactiva
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="🔋 Explorador de Predicciones Solar",
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
st.title("🔋 Explorador Interactivo de Predicciones Solares")
st.markdown("---")

# ===============================================================
# HORAS DE SOL REALES EN ESPAÑA
# ===============================================================

HORAS_SOL_MADRID = {
    1:  {'salida': 8.5, 'puesta': 18.0},   # Enero
    2:  {'salida': 8.0, 'puesta': 18.5},   # Febrero
    3:  {'salida': 7.0, 'puesta': 19.5},   # Marzo
    4:  {'salida': 7.5, 'puesta': 20.5},   # Abril
    5:  {'salida': 7.0, 'puesta': 21.0},   # Mayo
    6:  {'salida': 7.0, 'puesta': 21.5},   # Junio
    7:  {'salida': 7.0, 'puesta': 21.5},   # Julio
    8:  {'salida': 7.5, 'puesta': 21.0},   # Agosto
    9:  {'salida': 8.0, 'puesta': 20.0},   # Septiembre
    10: {'salida': 8.0, 'puesta': 19.5},   # Octubre
    11: {'salida': 8.0, 'puesta': 18.0},   # Noviembre
    12: {'salida': 8.5, 'puesta': 18.0},   # Diciembre
}

def hora_tiene_sol(hora, mes):
    """Determinar si una hora tiene sol según el mes en España"""
    if mes not in HORAS_SOL_MADRID:
        return False
    salida = HORAS_SOL_MADRID[mes]['salida']
    puesta = HORAS_SOL_MADRID[mes]['puesta']
    return salida <= hora < puesta

# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================

@st.cache_data
def cargar_datos():
    """Cargar datos originales"""
    df = pd.read_excel('Datos reales.xlsx', engine='openpyxl')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['fecha_y_hora'] = pd.to_datetime(df['fecha y hora'])
    df = df.sort_values('fecha_y_hora').reset_index(drop=True)
    return df

@st.cache_data
def cargar_resultados():
    """Cargar resultados del modelo"""
    try:
        resultados = pd.read_excel('RESULTADOS_CORREGIDOS_SOL.xlsx')
        resultados['fecha_y_hora'] = pd.to_datetime(resultados['fecha_y_hora'])
        return resultados
    except:
        return None

def preparar_features(df):
    """Preparar características para predicción"""
    df = df.copy()
    
    # Asegurar que fecha_y_hora es datetime
    if not pd.api.types.is_datetime64_any_dtype(df['fecha_y_hora']):
        df['fecha_y_hora'] = pd.to_datetime(df['fecha_y_hora'])
    
    # Temporales
    df['hora'] = df['fecha_y_hora'].dt.hour
    df['minuto'] = df['fecha_y_hora'].dt.minute
    
    # Crear columna fecha si no existe
    if 'fecha' not in df.columns:
        df['fecha'] = df['fecha_y_hora'].dt.date
    else:
        if not pd.api.types.is_datetime64_any_dtype(df['fecha']):
            df['fecha'] = pd.to_datetime(df['fecha'])
    
    df['dia_semana'] = df['fecha_y_hora'].dt.dayofweek
    df['mes'] = df['fecha_y_hora'].dt.month
    df['dia_año'] = df['fecha_y_hora'].dt.dayofyear
    df['es_finde'] = df['dia_semana'].isin([5, 6]).astype(int)
    df['trimestre'] = df['fecha_y_hora'].dt.quarter
    
    # Cíclicas
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    # Solares
    df['es_dia_solar'] = df['hora'].between(6, 18).astype(int)
    df['intensidad_solar'] = np.maximum(0, np.sin(np.pi * (df['hora'] - 6) / 12))
    
    # NUEVO: Tiene sol según mes (España)
    df['tiene_sol'] = df.apply(lambda row: int(hora_tiene_sol(row['hora'], row['mes'])), axis=1)
    
    return df

@st.cache_data
def cargar_modelo_entrenado():
    """Cargar el modelo previamente entrenado"""
    try:
        with open('modelo_solar_entrenado.pkl', 'rb') as f:
            modelo_info = pickle.load(f)
        return modelo_info['model'], modelo_info['feature_cols'], modelo_info['scaler_info']
    except FileNotFoundError:
        st.error("❌ No se encontró el modelo entrenado. Ejecuta primero 'modelo_corregido_horas_sol.py'")
        return None, None, None

def crear_modelo_simple_fallback(df, feature_cols):
    """Modelo simple de respaldo si no hay suficientes datos"""
    X = df[feature_cols].values
    y = df['generacion'].values
    
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X, y)
    return model, feature_cols

def predecir_futuro(model, feature_cols, df, scaler_info, dias_adelante=7):
    """Generar predicciones hacia el futuro con mejor lógica"""
    ultima_fecha = df['fecha_y_hora'].max()
    
    # Generar fechas futuras (cada 15 minutos)
    fechas_futuras = pd.date_range(
        start=ultima_fecha + timedelta(minutes=15),
        periods=dias_adelante * 24 * 4,
        freq='15T'
    )
    
    df_futuro = pd.DataFrame({'fecha_y_hora': fechas_futuras})
    df_futuro['fecha'] = df_futuro['fecha_y_hora'].dt.date
    df_futuro = preparar_features(df_futuro)
    
    # Calcular estadísticas históricas más sofisticadas
    df_prep = preparar_features(df.copy())
    promedios_hora = df_prep.groupby('hora')['generacion'].mean()
    promedios_hora_mes = df_prep.groupby(['hora', 'mes'])['generacion'].mean()
    promedios_hora_dia = df_prep.groupby(['hora', 'dia_semana'])['generacion'].mean()
    
    # Calcular medias móviles históricas
    medias_moviles_historicas = {}
    for window in [3, 6, 12, 24]:
        medias_moviles_historicas[window] = df_prep['generacion'].rolling(window=window).mean().mean()
    
    predicciones = []
    for idx, row in df_futuro.iterrows():
        hora = row['hora']
        mes = row['mes']
        dia_semana = row['dia_semana']
        
        # Crear features con mejor lógica
        features = []
        for col in feature_cols:
            if 'lag_' in col:
                lag_num = int(col.split('_')[1])
                # Usar promedio histórico apropiado
                if lag_num <= 24:
                    features.append(promedios_hora.get(hora, 0))
                else:
                    features.append(promedios_hora.get(hora, 0) * 0.8)  # Reducir para lags largos
            elif 'media_movil_' in col:
                window = int(col.split('_')[2])
                features.append(medias_moviles_historicas.get(window, 0))
            elif col in df_futuro.columns:
                features.append(row[col])
            else:
                features.append(0)
        
        pred = model.predict([features])[0]
        pred = max(0, pred)
        
        # CORRECCIÓN: Ajustar según horas de sol reales de España
        if not hora_tiene_sol(hora, mes):
            pred = 0
        else:
            # Ajustar según el patrón histórico de esa hora y mes
            factor_ajuste = promedios_hora_mes.get((hora, mes), promedios_hora.get(hora, 1))
            if factor_ajuste > 0:
                pred = pred * min(2.0, max(0.1, factor_ajuste / promedios_hora.get(hora, 1)))
        
        predicciones.append(pred)
    
    df_futuro['Prediccion'] = predicciones
    
    return df_futuro

# ===============================================================
# SIDEBAR - CONTROLES
# ===============================================================

st.sidebar.title("⚙️ Controles de Exploración")

# Cargar datos
df = cargar_datos()
resultados = cargar_resultados()

# Opciones de navegación
modo = st.sidebar.selectbox(
    "🔍 Modo de Exploración:",
    ["📊 Datos Históricos (Real vs Predicho)", 
     "🔮 Predicciones Futuras",
     "📅 Día Específico"]
)

st.sidebar.markdown("---")

# Información de horas de sol
with st.sidebar.expander("☀️ Horas de Sol en España (Madrid)"):
    st.markdown("### Horas de luz por mes:")
    for mes in range(1, 13):
        mes_nombre = ['', 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                      'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][mes]
        horas = HORAS_SOL_MADRID[mes]
        duracion = horas['puesta'] - horas['salida']
        st.markdown(f"**{mes_nombre}:** {horas['salida']:.1f}h - {horas['puesta']:.1f}h ({duracion:.1f}h)")
    
    st.info("🌍 Las predicciones se ajustan automáticamente según estas horas de sol")

st.sidebar.markdown("---")

# ===============================================================
# MODO 1: DATOS HISTÓRICOS
# ===============================================================

if modo == "📊 Datos Históricos (Real vs Predicho)":
    st.header("📊 Exploración de Datos Históricos")
    
    if resultados is not None:
        # Selector de rango de fechas
        col1, col2 = st.columns(2)
        
        with col1:
            fecha_inicio = st.date_input(
                "📅 Fecha Inicio:",
                value=resultados['fecha_y_hora'].min().date(),
                min_value=resultados['fecha_y_hora'].min().date(),
                max_value=resultados['fecha_y_hora'].max().date()
            )
        
        with col2:
            fecha_fin = st.date_input(
                "📅 Fecha Fin:",
                value=resultados['fecha_y_hora'].max().date(),
                min_value=resultados['fecha_y_hora'].min().date(),
                max_value=resultados['fecha_y_hora'].max().date()
            )
        
        # Filtrar datos
        mask = (resultados['fecha_y_hora'].dt.date >= fecha_inicio) & \
               (resultados['fecha_y_hora'].dt.date <= fecha_fin)
        datos_filtrados = resultados[mask].copy()
        
        if len(datos_filtrados) > 0:
            # Métricas
            st.markdown("### 📈 Métricas del Período Seleccionado")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mae = np.mean(np.abs(datos_filtrados['generacion'] - datos_filtrados['Prediccion_Corregida']))
                st.metric("MAE", f"{mae:.1f} kWh")
            
            with col2:
                r2 = 1 - (np.sum((datos_filtrados['generacion'] - datos_filtrados['Prediccion_Corregida'])**2) / 
                         np.sum((datos_filtrados['generacion'] - datos_filtrados['generacion'].mean())**2))
                st.metric("R²", f"{r2:.4f}")
            
            with col3:
                max_real = datos_filtrados['generacion'].max()
                st.metric("Max Real", f"{max_real:.0f} kWh")
            
            with col4:
                max_pred = datos_filtrados['Prediccion_Corregida'].max()
                st.metric("Max Predicho", f"{max_pred:.0f} kWh")
            
            st.markdown("---")
            
            # GRÁFICA 1: Superposición
            st.markdown("### 📊 Gráfica 1: Superposición Real vs Predicción")
            
            fig1 = go.Figure()
            
            fig1.add_trace(go.Scatter(
                x=datos_filtrados['fecha_y_hora'],
                y=datos_filtrados['generacion'],
                mode='lines',
                name='Real',
                line=dict(color='blue', width=2),
                hovertemplate='%{x}<br>Real: %{y:.0f} kWh<extra></extra>'
            ))
            
            fig1.add_trace(go.Scatter(
                x=datos_filtrados['fecha_y_hora'],
                y=datos_filtrados['Prediccion_Corregida'],
                mode='lines',
                name='Predicción',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='%{x}<br>Predicción: %{y:.0f} kWh<extra></extra>'
            ))
            
            fig1.update_layout(
                title="Superposición: Real vs Predicción",
                xaxis_title="Fecha y Hora",
                yaxis_title="Generación (kWh)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # GRÁFICA 2: Zoom (primeros 3 días)
            st.markdown("### 🔍 Gráfica 2: Zoom - Primeros 3 Días")
            
            n_registros_3dias = min(3 * 24 * 4, len(datos_filtrados))
            datos_zoom = datos_filtrados.iloc[:n_registros_3dias]
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=datos_zoom['fecha_y_hora'],
                y=datos_zoom['generacion'],
                mode='lines+markers',
                name='Real',
                line=dict(color='blue', width=3),
                marker=dict(size=4),
                hovertemplate='%{x}<br>Real: %{y:.0f} kWh<extra></extra>'
            ))
            
            fig2.add_trace(go.Scatter(
                x=datos_zoom['fecha_y_hora'],
                y=datos_zoom['Prediccion_Corregida'],
                mode='lines',
                name='Predicción',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='%{x}<br>Predicción: %{y:.0f} kWh<extra></extra>'
            ))
            
            fig2.update_layout(
                title="Zoom: Detalle de los Primeros 3 Días",
                xaxis_title="Fecha y Hora",
                yaxis_title="Generación (kWh)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # GRÁFICAS 3 y 4 en columnas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Gráfica 3: Scatter Real vs Predicción")
                
                fig3 = go.Figure()
                
                fig3.add_trace(go.Scatter(
                    x=datos_filtrados['generacion'],
                    y=datos_filtrados['Prediccion_Corregida'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=datos_filtrados['hora'],
                        colorscale='Rainbow',
                        showscale=True,
                        colorbar=dict(title="Hora")
                    ),
                    text=datos_filtrados['hora'],
                    hovertemplate='Real: %{x:.0f} kWh<br>Predicción: %{y:.0f} kWh<br>Hora: %{text}<extra></extra>'
                ))
                
                # Línea ideal
                max_val = max(datos_filtrados['generacion'].max(), datos_filtrados['Prediccion_Corregida'].max())
                fig3.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Ideal',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig3.update_layout(
                    title=f"Predicción vs Real (R²={r2:.4f})",
                    xaxis_title="Real (kWh)",
                    yaxis_title="Predicción (kWh)",
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Gráfica 4: Patrón Diario Promedio")
                
                patron_real = datos_filtrados.groupby('hora')['generacion'].mean()
                patron_pred = datos_filtrados.groupby('hora')['Prediccion_Corregida'].mean()
                patron_std = datos_filtrados.groupby('hora')['generacion'].std()
                
                fig4 = go.Figure()
                
                # Banda de desviación estándar
                fig4.add_trace(go.Scatter(
                    x=patron_real.index,
                    y=patron_real + patron_std,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig4.add_trace(go.Scatter(
                    x=patron_real.index,
                    y=patron_real - patron_std,
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(0, 100, 255, 0.2)',
                    fill='tonexty',
                    name='Desv. Estándar',
                    hoverinfo='skip'
                ))
                
                fig4.add_trace(go.Scatter(
                    x=patron_real.index,
                    y=patron_real.values,
                    mode='lines+markers',
                    name='Real',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10, symbol='circle'),
                    hovertemplate='Hora %{x}<br>Real: %{y:.0f} kWh<extra></extra>'
                ))
                
                fig4.add_trace(go.Scatter(
                    x=patron_pred.index,
                    y=patron_pred.values,
                    mode='lines+markers',
                    name='Predicción',
                    line=dict(color='red', width=3),
                    marker=dict(size=10, symbol='square'),
                    hovertemplate='Hora %{x}<br>Predicción: %{y:.0f} kWh<extra></extra>'
                ))
                
                fig4.update_layout(
                    title="Patrón Diario Promedio",
                    xaxis_title="Hora del día",
                    yaxis_title="Generación promedio (kWh)",
                    height=500,
                    template='plotly_white',
                    xaxis=dict(tickmode='linear', tick0=0, dtick=2)
                )
                
                st.plotly_chart(fig4, use_container_width=True)
            
            # Tabla de datos
            with st.expander("📋 Ver Datos Detallados"):
                st.dataframe(
                    datos_filtrados[['fecha_y_hora', 'hora', 'generacion', 'Prediccion_Corregida', 'Error']].head(100),
                    use_container_width=True
                )
        
        else:
            st.warning("⚠️ No hay datos en el rango seleccionado")
    
    else:
        st.info("ℹ️ No hay resultados previos. Primero ejecuta 'modelo_corregido_horas_sol.py'")

# ===============================================================
# MODO 2: PREDICCIONES FUTURAS
# ===============================================================

elif modo == "🔮 Predicciones Futuras":
    st.header("🔮 Predicciones Futuras")
    
    st.success("✅ **PREDICCIONES CORREGIDAS:** Respetan las horas de sol reales de España por mes")
    st.info("☀️ En enero: solo predice 8:30-18:00 | En junio: predice 7:00-21:30")
    
    # Cargar modelo entrenado
    with st.spinner("🔄 Cargando modelo entrenado..."):
        model, feature_cols, scaler_info = cargar_modelo_entrenado()
    
    if model is None:
        st.stop()
    
    st.success("✅ Modelo cargado correctamente")
    
    # Selector de días hacia adelante
    dias_adelante = st.slider(
        "📅 Días hacia el futuro:",
        min_value=1,
        max_value=30,
        value=7,
        step=1
    )
    
    # Generar predicciones
    with st.spinner(f"🔮 Generando predicciones para {dias_adelante} días..."):
        df_futuro = predecir_futuro(model, feature_cols, df, scaler_info, dias_adelante)
    
    st.success(f"✅ Predicciones generadas para {dias_adelante} días")
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_predicho = df_futuro['Prediccion'].sum()
        st.metric("Total Predicho", f"{total_predicho:,.0f} kWh")
    
    with col2:
        max_predicho = df_futuro['Prediccion'].max()
        st.metric("Máximo Predicho", f"{max_predicho:,.0f} kWh")
    
    with col3:
        promedio = df_futuro['Prediccion'].mean()
        st.metric("Promedio", f"{promedio:,.0f} kWh")
    
    st.markdown("---")
    
    # Gráfica de predicciones futuras
    st.markdown("### 📈 Predicciones Futuras")
    
    fig_futuro = go.Figure()
    
    # Últimos días históricos
    ultimos_dias = df.tail(7 * 24 * 4)[['fecha_y_hora', 'generacion']].copy()
    
    fig_futuro.add_trace(go.Scatter(
        x=ultimos_dias['fecha_y_hora'],
        y=ultimos_dias['generacion'],
        mode='lines',
        name='Histórico',
        line=dict(color='blue', width=2),
        hovertemplate='%{x}<br>Histórico: %{y:.0f} kWh<extra></extra>'
    ))
    
    fig_futuro.add_trace(go.Scatter(
        x=df_futuro['fecha_y_hora'],
        y=df_futuro['Prediccion'],
        mode='lines',
        name='Predicción',
        line=dict(color='orange', width=3),
        hovertemplate='%{x}<br>Predicción: %{y:.0f} kWh<extra></extra>'
    ))
    
    fig_futuro.update_layout(
        title=f"Predicciones para los próximos {dias_adelante} días",
        xaxis_title="Fecha y Hora",
        yaxis_title="Generación (kWh)",
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_futuro, use_container_width=True)
    
    # Patrón diario futuro
    st.markdown("### 📊 Patrón Diario Predicho")
    
    patron_futuro = df_futuro.groupby('hora')['Prediccion'].mean()
    
    fig_patron = go.Figure()
    
    fig_patron.add_trace(go.Bar(
        x=patron_futuro.index,
        y=patron_futuro.values,
        marker_color='orange',
        hovertemplate='Hora %{x}<br>Predicción: %{y:.0f} kWh<extra></extra>'
    ))
    
    fig_patron.update_layout(
        title="Patrón Diario Promedio Predicho",
        xaxis_title="Hora del día",
        yaxis_title="Generación promedio (kWh)",
        height=400,
        template='plotly_white',
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    
    st.plotly_chart(fig_patron, use_container_width=True)
    
    # Descargar predicciones
    csv = df_futuro[['fecha_y_hora', 'Prediccion']].to_csv(index=False)
    st.download_button(
        label="📥 Descargar Predicciones (CSV)",
        data=csv,
        file_name=f"predicciones_{dias_adelante}dias.csv",
        mime="text/csv"
    )

# ===============================================================
# MODO 3: DÍA ESPECÍFICO
# ===============================================================

elif modo == "📅 Día Específico":
    st.header("📅 Exploración de Día Específico")
    
    if resultados is not None:
        # Selector de fecha
        fecha_seleccionada = st.date_input(
            "📅 Selecciona un día:",
            value=resultados['fecha_y_hora'].min().date(),
            min_value=resultados['fecha_y_hora'].min().date(),
            max_value=resultados['fecha_y_hora'].max().date()
        )
        
        # Filtrar datos del día
        datos_dia = resultados[resultados['fecha_y_hora'].dt.date == fecha_seleccionada].copy()
        
        if len(datos_dia) > 0:
            # Métricas del día
            st.markdown(f"### 📊 Métricas del {fecha_seleccionada}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_real = datos_dia['generacion'].sum()
                st.metric("Total Real", f"{total_real:.0f} kWh")
            
            with col2:
                total_pred = datos_dia['Prediccion_Corregida'].sum()
                st.metric("Total Predicho", f"{total_pred:.0f} kWh")
            
            with col3:
                error_dia = total_real - total_pred
                st.metric("Diferencia", f"{error_dia:+.0f} kWh")
            
            with col4:
                error_pct = (error_dia / total_real * 100) if total_real > 0 else 0
                st.metric("Error %", f"{error_pct:+.1f}%")
            
            st.markdown("---")
            
            # Gráfica del día
            st.markdown("### 📈 Generación del Día")
            
            fig_dia = go.Figure()
            
            fig_dia.add_trace(go.Scatter(
                x=datos_dia['fecha_y_hora'],
                y=datos_dia['generacion'],
                mode='lines+markers',
                name='Real',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                hovertemplate='%{x}<br>Real: %{y:.0f} kWh<extra></extra>'
            ))
            
            fig_dia.add_trace(go.Scatter(
                x=datos_dia['fecha_y_hora'],
                y=datos_dia['Prediccion_Corregida'],
                mode='lines+markers',
                name='Predicción',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8, symbol='square'),
                hovertemplate='%{x}<br>Predicción: %{y:.0f} kWh<extra></extra>'
            ))
            
            fig_dia.update_layout(
                title=f"Comparación: Real vs Predicción - {fecha_seleccionada}",
                xaxis_title="Hora",
                yaxis_title="Generación (kWh)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_dia, use_container_width=True)
            
            # Tabla horaria
            st.markdown("### 📋 Datos Horarios")
            
            tabla_dia = datos_dia[['fecha_y_hora', 'hora', 'generacion', 'Prediccion_Corregida', 'Error']].copy()
            tabla_dia.columns = ['Fecha y Hora', 'Hora', 'Real (kWh)', 'Predicción (kWh)', 'Error (kWh)']
            
            st.dataframe(tabla_dia, use_container_width=True)
        
        else:
            st.warning(f"⚠️ No hay datos para {fecha_seleccionada}")
    
    else:
        st.info("ℹ️ No hay resultados previos. Primero ejecuta 'modelo_corregido_horas_sol.py'")

# Footer
st.markdown("---")
st.markdown("🔋 **Explorador Interactivo de Predicciones** | Desarrollado con Streamlit y Gradient Boosting")
st.markdown("☀️ **Versión 2.0** - Corregido con horas de sol reales de España | Predicciones ajustadas por mes")

