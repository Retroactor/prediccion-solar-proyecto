# ===============================================================
# 🔧 PREPROCESS_DATA - TRANSFORMACIONES COMUNES
# ===============================================================
# Archivo para realizar transformaciones, limpiezas de datos
# y cambios requeridos comunes para todos los modelos
# ===============================================================

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_solar_data(df):
    """
    Limpiar datos solares comunes
    
    Args:
        df (pd.DataFrame): DataFrame con datos solares
        
    Returns:
        pd.DataFrame: DataFrame limpio
    """
    print("🧹 Iniciando limpieza de datos...")
    
    # Crear copia para no modificar el original
    df_clean = df.copy()
    
    # 1. Limpiar nombres de columnas
    df_clean.columns = df_clean.columns.str.strip()
    
    # 2. Convertir fecha_y_hora a datetime si existe
    fecha_col = None
    if 'fecha_y_hora' in df_clean.columns:
        fecha_col = 'fecha_y_hora'
    elif 'fecha y hora' in df_clean.columns:
        fecha_col = 'fecha y hora'
        # Renombrar para consistencia
        df_clean = df_clean.rename(columns={'fecha y hora': 'fecha_y_hora'})
    
    if fecha_col or 'fecha_y_hora' in df_clean.columns:
        df_clean['fecha_y_hora'] = pd.to_datetime(df_clean['fecha_y_hora'])
        print("✅ Fecha convertida a datetime")
    
    # 3. Ordenar por fecha si existe
    if 'fecha_y_hora' in df_clean.columns:
        df_clean = df_clean.sort_values('fecha_y_hora').reset_index(drop=True)
        print("✅ Datos ordenados por fecha")
    
    # 4. Limpiar valores de generación
    if 'generacion' in df_clean.columns:
        # Eliminar valores negativos
        df_clean = df_clean[df_clean['generacion'] >= 0]
        
        # Eliminar outliers extremos (valores > 99% percentil)
        q99 = df_clean['generacion'].quantile(0.99)
        df_clean = df_clean[df_clean['generacion'] <= q99 * 1.1]  # Permitir 10% más que el percentil 99
        
        print(f"✅ Generación limpia: {len(df_clean)} registros válidos")
    
    # 5. Limpiar valores de radiación si existe
    if 'Total Promedio de Radiación inclinada Solar R1(W/m²)' in df_clean.columns:
        col_rad = 'Total Promedio de Radiación inclinada Solar R1(W/m²)'
        # Eliminar valores negativos
        df_clean = df_clean[df_clean[col_rad] >= 0]
        
        # Eliminar outliers extremos
        q99_rad = df_clean[col_rad].quantile(0.99)
        df_clean = df_clean[df_clean[col_rad] <= q99_rad * 1.2]
        
        print(f"✅ Radiación limpia: outliers removidos")
    
    # 6. Limpiar temperatura si existe
    if 'Total Promedio de Temperatura del módulo 1(°C)' in df_clean.columns:
        col_temp = 'Total Promedio de Temperatura del módulo 1(°C)'
        # Eliminar temperaturas imposibles (< -50°C o > 100°C)
        df_clean = df_clean[
            (df_clean[col_temp] >= -50) & 
            (df_clean[col_temp] <= 100)
        ]
        
        print(f"✅ Temperatura limpia: valores imposibles removidos")
    
    # 7. Eliminar filas con demasiados NaN
    threshold_nan = 0.5  # Máximo 50% de valores NaN por fila
    df_clean = df_clean.dropna(thresh=int(len(df_clean.columns) * (1 - threshold_nan)))
    
    # 8. Resetear índice
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"✅ Limpieza completada: {len(df_clean)} registros finales")
    
    return df_clean

def validate_data_structure(df):
    """
    Validar estructura de datos requerida
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        
    Returns:
        bool: True si la estructura es válida
    """
    print("🔍 Validando estructura de datos...")
    
    # Verificar columnas requeridas (con flexibilidad en nombres)
    has_fecha = 'fecha_y_hora' in df.columns or 'fecha y hora' in df.columns
    has_generacion = 'generacion' in df.columns
    
    if not has_fecha:
        print("❌ Columna de fecha faltante (necesita 'fecha_y_hora' o 'fecha y hora')")
        return False
    
    if not has_generacion:
        print("❌ Columna 'generacion' faltante")
        return False
    
    # Validar tipos de datos
    fecha_col_name = 'fecha_y_hora' if 'fecha_y_hora' in df.columns else 'fecha y hora'
    if not pd.api.types.is_datetime64_any_dtype(df[fecha_col_name]):
        print(f"❌ '{fecha_col_name}' debe ser de tipo datetime")
        return False
    
    if not pd.api.types.is_numeric_dtype(df['generacion']):
        print("❌ 'generacion' debe ser numérica")
        return False
    
    # Validar rango de fechas
    fecha_col_name = 'fecha_y_hora' if 'fecha_y_hora' in df.columns else 'fecha y hora'
    fecha_min = df[fecha_col_name].min()
    fecha_max = df[fecha_col_name].max()
    
    if fecha_min >= fecha_max:
        print("❌ Rango de fechas inválido")
        return False
    
    print("✅ Estructura de datos válida")
    print(f"📅 Rango de fechas: {fecha_min} a {fecha_max}")
    print(f"📊 Total de registros: {len(df)}")
    
    return True

def add_basic_features(df):
    """
    Agregar features básicas comunes
    
    Args:
        df (pd.DataFrame): DataFrame base
        
    Returns:
        pd.DataFrame: DataFrame con features básicas
    """
    print("🔧 Agregando features básicas...")
    
    df_features = df.copy()
    
    # Features temporales básicas
    df_features['año'] = df_features['fecha_y_hora'].dt.year
    df_features['mes'] = df_features['fecha_y_hora'].dt.month
    df_features['dia'] = df_features['fecha_y_hora'].dt.day
    df_features['hora'] = df_features['fecha_y_hora'].dt.hour
    df_features['minuto'] = df_features['fecha_y_hora'].dt.minute
    df_features['dia_semana'] = df_features['fecha_y_hora'].dt.dayofweek
    df_features['dia_año'] = df_features['fecha_y_hora'].dt.dayofyear
    df_features['semana_año'] = df_features['fecha_y_hora'].dt.isocalendar().week
    
    # Features derivadas
    df_features['es_finde'] = df_features['dia_semana'].isin([5, 6]).astype(int)
    df_features['es_verano'] = df_features['mes'].isin([6, 7, 8]).astype(int)
    df_features['es_invierno'] = df_features['mes'].isin([12, 1, 2]).astype(int)
    
    print("✅ Features básicas agregadas")
    
    return df_features

def split_train_test(df, test_size=0.2, random_state=42):
    """
    Dividir datos en entrenamiento y prueba de forma temporal
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        test_size (float): Proporción de datos para prueba
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (df_train, df_test)
    """
    print(f"✂️ Dividiendo datos: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    
    # Ordenar por fecha para split temporal
    df_sorted = df.sort_values('fecha_y_hora').reset_index(drop=True)
    
    # Split temporal (últimos datos para test)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    df_train = df_sorted.iloc[:split_idx].copy()
    df_test = df_sorted.iloc[split_idx:].copy()
    
    print(f"📊 Train: {len(df_train)} registros")
    print(f"📊 Test: {len(df_test)} registros")
    print(f"📅 Train: {df_train['fecha_y_hora'].min()} a {df_train['fecha_y_hora'].max()}")
    print(f"📅 Test: {df_test['fecha_y_hora'].min()} a {df_test['fecha_y_hora'].max()}")
    
    return df_train, df_test

def preprocess_data(input_path, output_path):
    """
    Función principal de preprocesamiento
    
    Args:
        input_path (str): Ruta del archivo de datos en crudo
        output_path (str): Ruta donde guardar los datos procesados
    """
    print("=" * 60)
    print("🔧 PREPROCESAMIENTO DE DATOS SOLARES")
    print("=" * 60)
    
    try:
        # 1. Cargar datos
        print(f"📂 Cargando datos desde: {input_path}")
        
        if input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        elif input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        else:
            raise ValueError("❌ Formato de archivo no soportado. Use .xlsx o .csv")
        
        print(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        
        # 2. Validar estructura
        if not validate_data_structure(df):
            raise ValueError("❌ Estructura de datos inválida")
        
        # 3. Limpiar datos
        df_clean = clean_solar_data(df)
        
        # 4. Agregar features básicas
        df_features = add_basic_features(df_clean)
        
        # 5. Guardar datos procesados
        print(f"💾 Guardando datos procesados en: {output_path}")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar en formato Excel
        df_features.to_excel(output_path, index=False)
        
        print("✅ Preprocesamiento completado exitosamente")
        print(f"📊 Datos finales: {len(df_features)} registros, {len(df_features.columns)} columnas")
        
        return df_features
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Ejemplo de uso
    if len(sys.argv) != 3:
        print("Uso: python preprocess_data.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    preprocess_data(input_path, output_path)
