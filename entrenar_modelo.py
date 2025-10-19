# ===============================================================
# 🚀 TRAIN_MODELS - ENTRENAMIENTO DE MODELOS
# ===============================================================
# Script para entrenar modelos usando la arquitectura profesional
# Importa el paquete, instancia modelos, los entrena y serializa
# ===============================================================

import os
import sys
import configparser
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar nuestras clases
from modelo_solar_predictor import SolarPredictor
from procesar_datos import preprocess_data, split_train_test

def load_config(config_path):
    """
    Cargar configuración desde archivo .conf
    
    Args:
        config_path (str): Ruta del archivo de configuración
        
    Returns:
        configparser.ConfigParser: Configuración cargada
    """
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Archivo de configuración no encontrado: {config_path}")
    
    config.read(config_path)
    print(f"✅ Configuración cargada desde: {config_path}")
    
    return config

def create_directories(config):
    """Crear directorios necesarios"""
    model_path = config['PATHS']['model_output_path']
    results_path = config['PATHS']['results_output_path']
    
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print(f"📁 Directorios creados: {model_path}, {results_path}, data/")

def load_model_params(config):
    """
    Cargar parámetros del modelo desde configuración
    
    Args:
        config: Configuración cargada
        
    Returns:
        dict: Parámetros del modelo
    """
    model_params = {}
    
    # Parámetros del modelo
    model_params['n_estimators'] = config.getint('MODEL_PARAMS', 'n_estimators')
    model_params['max_depth'] = config.getint('MODEL_PARAMS', 'max_depth')
    model_params['learning_rate'] = config.getfloat('MODEL_PARAMS', 'learning_rate')
    model_params['subsample'] = config.getfloat('MODEL_PARAMS', 'subsample')
    model_params['random_state'] = config.getint('MODEL_PARAMS', 'random_state')
    
    print("🔧 Parámetros del modelo cargados:")
    for key, value in model_params.items():
        print(f"   {key}: {value}")
    
    return model_params

def train_solar_predictor(config):
    """
    Entrenar modelo SolarPredictor
    
    Args:
        config: Configuración cargada
        
    Returns:
        tuple: (model, metrics)
    """
    print("=" * 60)
    print("🌅 ENTRENANDO SOLAR PREDICTOR")
    print("=" * 60)
    
    # 1. Cargar parámetros
    model_params = load_model_params(config)
    
    # 2. Rutas de datos
    raw_data_path = config['PATHS']['raw_data_path']
    processed_data_path = config['PATHS']['processed_data_path']
    
    # 3. Preprocesar datos si es necesario
    if not os.path.exists(processed_data_path):
        print("🔧 Preprocesando datos...")
        preprocess_data(raw_data_path, processed_data_path)
    else:
        print(f"✅ Datos procesados ya existen: {processed_data_path}")
    
    # 4. Cargar datos procesados
    print("📂 Cargando datos procesados...")
    df = pd.read_excel(processed_data_path)
    print(f"✅ Datos cargados: {len(df)} registros")
    
    # 5. Dividir datos
    test_size = config.getfloat('TRAINING_PARAMS', 'test_size')
    random_state = config.getint('TRAINING_PARAMS', 'random_state')
    
    df_train, df_test = split_train_test(df, test_size, random_state)
    
    # 6. Crear y entrenar modelo
    print("🚀 Creando modelo SolarPredictor...")
    model = SolarPredictor(model_params)
    
    print("🔄 Entrenando modelo...")
    model.fit(df_train)
    
    # 7. Evaluar modelo
    print("📊 Evaluando modelo...")
    y_test = df_test['generacion'].values
    X_test = df_test.drop('generacion', axis=1) if 'generacion' in df_test.columns else df_test
    
    metrics = model.evaluate(X_test, y_test)
    
    print("📈 Métricas de evaluación:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # 8. Guardar modelo
    model_path = config['PATHS']['model_output_path']
    model_filename = os.path.join(model_path, 'solar_predictor_model.pkl')
    
    print(f"💾 Guardando modelo en: {model_filename}")
    model.save(model_filename)
    
    # 9. Guardar métricas
    if config.getboolean('OUTPUT', 'save_metrics'):
        metrics_path = os.path.join(config['PATHS']['results_output_path'], 'training_metrics.txt')
        
        with open(metrics_path, 'w') as f:
            f.write("MÉTRICAS DE ENTRENAMIENTO - SOLAR PREDICTOR\n")
            f.write("=" * 50 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Datos de entrenamiento: {len(df_train)} registros\n")
            f.write(f"Datos de prueba: {len(df_test)} registros\n")
            f.write(f"Features utilizadas: {len(model.feature_cols)}\n\n")
            
            f.write("MÉTRICAS:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nPARÁMETROS DEL MODELO:\n")
            for param, value in model_params.items():
                f.write(f"{param}: {value}\n")
        
        print(f"📊 Métricas guardadas en: {metrics_path}")
    
    # 10. Generar predicciones de ejemplo
    if config.getboolean('OUTPUT', 'save_predictions'):
        print("🔮 Generando predicciones de ejemplo...")
        
        # Tomar una muestra de datos de prueba
        sample_size = min(100, len(df_test))
        df_sample = df_test.head(sample_size)
        X_sample = df_sample.drop('generacion', axis=1) if 'generacion' in df_sample.columns else df_sample
        
        predictions = model.predict(X_sample)
        
        # Crear DataFrame con resultados
        results_df = df_sample.copy()
        results_df['prediccion'] = predictions
        results_df['error'] = np.abs(results_df['generacion'] - predictions)
        
        # Guardar resultados
        results_path = os.path.join(config['PATHS']['results_output_path'], 'predicciones_ejemplo.xlsx')
        results_df.to_excel(results_path, index=False)
        
        print(f"🔮 Predicciones guardadas en: {results_path}")
    
    return model, metrics

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenar modelos de predicción solar')
    parser.add_argument('config_path', help='Ruta del archivo de configuración train.conf')
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS")
        print("=" * 60)
        
        # 1. Cargar configuración
        config = load_config(args.config_path)
        
        # 2. Crear directorios
        create_directories(config)
        
        # 3. Entrenar modelos
        model, metrics = train_solar_predictor(config)
        
        print("=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📊 R² Score: {metrics['R2']:.4f}")
        print(f"📊 MAE: {metrics['MAE']:.2f} kWh")
        print(f"📊 RMSE: {metrics['RMSE']:.2f} kWh")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
