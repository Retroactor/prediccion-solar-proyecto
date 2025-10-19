# ===============================================================
# 🔮 INFERENCE_MODEL - INFERENCIA DE MODELOS
# ===============================================================
# Script para cargar modelos entrenados y realizar inferencia
# en nuevos datos usando argparse para argumentos
# ===============================================================

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar nuestras clases
from modelo_solar_predictor import SolarPredictor

def load_model(model_type, model_path):
    """
    Cargar modelo entrenado
    
    Args:
        model_type (str): Tipo de modelo ('modelo_solar_predictor')
        model_path (str): Ruta del archivo del modelo
        
    Returns:
        SolarPredictor: Modelo cargado
    """
    if model_type.lower() == 'modelo_solar_predictor':
        model = SolarPredictor()
        model.load(model_path)
        return model
    else:
        raise ValueError(f"❌ Tipo de modelo no soportado: {model_type}")

def load_input_data(input_path):
    """
    Cargar datos de entrada
    
    Args:
        input_path (str): Ruta del archivo de datos
        
    Returns:
        pd.DataFrame: Datos cargados
    """
    print(f"📂 Cargando datos desde: {input_path}")
    
    if input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError("❌ Formato de archivo no soportado. Use .xlsx o .csv")
    
    print(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
    
    return df

def generate_future_predictions(model, start_date, days_ahead, output_path):
    """
    Generar predicciones futuras
    
    Args:
        model: Modelo entrenado
        start_date (str): Fecha de inicio en formato YYYY-MM-DD
        days_ahead (int): Días hacia adelante
        output_path (str): Ruta de salida
    """
    print(f"🔮 Generando predicciones para {days_ahead} días desde {start_date}")
    
    # Crear fechas futuras (cada 15 minutos)
    start_datetime = pd.to_datetime(start_date)
    fechas_futuras = pd.date_range(
        start=start_datetime,
        periods=days_ahead * 24 * 4,  # 4 registros por hora (cada 15 min)
        freq='15T'
    )
    
    # Crear DataFrame con fechas futuras
    df_futuro = pd.DataFrame({'fecha_y_hora': fechas_futuras})
    
    # Hacer predicciones
    predictions = model.predict(df_futuro)
    
    # Crear DataFrame con resultados
    df_results = df_futuro.copy()
    df_results['prediccion'] = predictions
    df_results['hora'] = df_results['fecha_y_hora'].dt.hour
    df_results['mes'] = df_results['fecha_y_hora'].dt.month
    df_results['dia_semana'] = df_results['fecha_y_hora'].dt.dayofweek
    
    # Guardar resultados
    df_results.to_excel(output_path, index=False)
    
    print(f"✅ Predicciones guardadas en: {output_path}")
    print(f"📊 Total de predicciones: {len(predictions)}")
    print(f"📈 Máxima predicción: {predictions.max():.2f} kWh")
    print(f"📈 Promedio: {predictions.mean():.2f} kWh")
    
    return df_results

def analyze_predictions(df_results):
    """
    Analizar predicciones generadas
    
    Args:
        df_results (pd.DataFrame): DataFrame con predicciones
        
    Returns:
        dict: Análisis de predicciones
    """
    print("📊 Analizando predicciones...")
    
    analysis = {}
    
    # Análisis por hora
    hourly_stats = df_results.groupby('hora')['prediccion'].agg(['mean', 'max', 'sum']).round(2)
    analysis['hourly_stats'] = hourly_stats
    
    # Análisis por día
    df_results['dia'] = df_results['fecha_y_hora'].dt.date
    daily_stats = df_results.groupby('dia')['prediccion'].agg(['sum', 'mean', 'max']).round(2)
    analysis['daily_stats'] = daily_stats
    
    # Análisis por mes
    monthly_stats = df_results.groupby('mes')['prediccion'].agg(['mean', 'max', 'sum']).round(2)
    analysis['monthly_stats'] = monthly_stats
    
    # Estadísticas generales
    analysis['general_stats'] = {
        'total_generation': df_results['prediccion'].sum(),
        'max_generation': df_results['prediccion'].max(),
        'avg_generation': df_results['prediccion'].mean(),
        'non_zero_predictions': (df_results['prediccion'] > 0).sum(),
        'zero_predictions': (df_results['prediccion'] == 0).sum()
    }
    
    print("📈 Análisis completado:")
    print(f"   Total generación: {analysis['general_stats']['total_generation']:.2f} kWh")
    print(f"   Máxima generación: {analysis['general_stats']['max_generation']:.2f} kWh")
    print(f"   Promedio: {analysis['general_stats']['avg_generation']:.2f} kWh")
    print(f"   Predicciones > 0: {analysis['general_stats']['non_zero_predictions']}")
    print(f"   Predicciones = 0: {analysis['general_stats']['zero_predictions']}")
    
    return analysis

def save_analysis(analysis, output_path):
    """
    Guardar análisis en archivo de texto
    
    Args:
        analysis (dict): Análisis de predicciones
        output_path (str): Ruta de salida
    """
    analysis_path = output_path.replace('.xlsx', '_analysis.txt')
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS DE PREDICCIONES SOLARES\n")
        f.write("=" * 50 + "\n")
        f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ESTADÍSTICAS GENERALES:\n")
        f.write("-" * 30 + "\n")
        for key, value in analysis['general_stats'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nESTADÍSTICAS POR HORA:\n")
        f.write("-" * 30 + "\n")
        f.write(analysis['hourly_stats'].to_string())
        
        f.write("\n\nESTADÍSTICAS POR DÍA:\n")
        f.write("-" * 30 + "\n")
        f.write(analysis['daily_stats'].to_string())
        
        f.write("\n\nESTADÍSTICAS POR MES:\n")
        f.write("-" * 30 + "\n")
        f.write(analysis['monthly_stats'].to_string())
    
    print(f"📊 Análisis guardado en: {analysis_path}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Realizar inferencia con modelos de predicción solar')
    parser.add_argument('model_type', help='Tipo de modelo (modelo_solar_predictor)')
    parser.add_argument('model_path', help='Ruta del archivo del modelo (.pkl)')
    parser.add_argument('input_data', help='Ruta del archivo de datos de entrada')
    parser.add_argument('--output', '-o', help='Ruta de salida (opcional)', default='predicciones_output.xlsx')
    parser.add_argument('--future', '-f', action='store_true', help='Generar predicciones futuras')
    parser.add_argument('--start-date', help='Fecha de inicio para predicciones futuras (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=7, help='Días hacia adelante para predicciones futuras')
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("🔮 INICIANDO INFERENCIA DE MODELOS")
        print("=" * 60)
        
        # 1. Cargar modelo
        print(f"📦 Cargando modelo: {args.model_type}")
        model = load_model(args.model_type, args.model_path)
        
        if args.future:
            # 2. Generar predicciones futuras
            if not args.start_date:
                # Usar fecha actual como inicio
                start_date = datetime.now().strftime('%Y-%m-%d')
            else:
                start_date = args.start_date
            
            df_results = generate_future_predictions(
                model, start_date, args.days, args.output
            )
            
            # 3. Analizar predicciones
            analysis = analyze_predictions(df_results)
            
            # 4. Guardar análisis
            save_analysis(analysis, args.output)
            
        else:
            # 2. Cargar datos de entrada
            df_input = load_input_data(args.input_data)
            
            # 3. Hacer predicciones
            print("🔮 Realizando predicciones...")
            predictions = model.predict(df_input)
            
            # 4. Crear DataFrame con resultados
            df_results = df_input.copy()
            df_results['prediccion'] = predictions
            
            if 'generacion' in df_input.columns:
                df_results['error'] = np.abs(df_input['generacion'] - predictions)
                df_results['error_pct'] = (df_results['error'] / df_input['generacion'] * 100).round(2)
            
            # 5. Guardar resultados
            df_results.to_excel(args.output, index=False)
            print(f"✅ Resultados guardados en: {args.output}")
        
        print("=" * 60)
        print("✅ INFERENCIA COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error durante la inferencia: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
