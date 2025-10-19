# ===============================================================
# 🌅 MODELO CORREGIDO - HORAS DE SOL REALES EN ESPAÑA
# ===============================================================
# Solución al problema: Picos extraños a las 19:00 en invierno
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
import sys
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 80)
print("🌅 MODELO CORREGIDO CON HORAS DE SOL REALES")
print("=" * 80)

# ===============================================================
# HORAS DE SOL REALES EN ESPAÑA (Madrid)
# ===============================================================

HORAS_SOL_MADRID = {
    1:  {'salida': 8.5, 'puesta': 18.0},   # Enero - ¡AQUÍ ESTÁ EL PROBLEMA!
    2:  {'salida': 8.0, 'puesta': 18.5},   # Febrero
    3:  {'salida': 7.0, 'puesta': 19.5},   # Marzo (cambio hora)
    4:  {'salida': 7.5, 'puesta': 20.5},   # Abril
    5:  {'salida': 7.0, 'puesta': 21.0},   # Mayo
    6:  {'salida': 7.0, 'puesta': 21.5},   # Junio (días más largos)
    7:  {'salida': 7.0, 'puesta': 21.5},   # Julio
    8:  {'salida': 7.5, 'puesta': 21.0},   # Agosto
    9:  {'salida': 8.0, 'puesta': 20.0},   # Septiembre
    10: {'salida': 8.0, 'puesta': 19.5},   # Octubre (cambio hora)
    11: {'salida': 8.0, 'puesta': 18.0},   # Noviembre
    12: {'salida': 8.5, 'puesta': 18.0},   # Diciembre (días más cortos)
}

def hora_tiene_sol(hora, mes):
    """Determinar si una hora tiene sol según el mes"""
    if mes not in HORAS_SOL_MADRID:
        return False
    
    salida = HORAS_SOL_MADRID[mes]['salida']
    puesta = HORAS_SOL_MADRID[mes]['puesta']
    
    return salida <= hora < puesta

# ===============================================================
# CARGAR DATOS
# ===============================================================

print("\n📊 Cargando datos...")
df = pd.read_excel('Datos reales.xlsx', engine='openpyxl')

df['fecha'] = pd.to_datetime(df['fecha'])
df['fecha_y_hora'] = pd.to_datetime(df['fecha y hora'])
df = df.sort_values('fecha_y_hora').reset_index(drop=True)

df['hora'] = df['fecha_y_hora'].dt.hour
df['minuto'] = df['fecha_y_hora'].dt.minute
df['dia_semana'] = df['fecha'].dt.dayofweek
df['mes'] = df['fecha'].dt.month
df['dia_año'] = df['fecha'].dt.dayofyear

# NUEVA CARACTERÍSTICA: Tiene sol según mes
df['tiene_sol'] = df.apply(lambda row: int(hora_tiene_sol(row['hora'], row['mes'])), axis=1)

print(f"✅ Datos cargados: {len(df)} registros")

# Mostrar ejemplo de cómo funciona
print("\n🌞 Ejemplo: ¿Tiene sol a diferentes horas en enero vs junio?")
print("-" * 60)
for hora in [6, 8, 12, 18, 19, 20]:
    sol_enero = "✅ SÍ" if hora_tiene_sol(hora, 1) else "❌ NO"
    sol_junio = "✅ SÍ" if hora_tiene_sol(hora, 6) else "❌ NO"
    print(f"   {hora:2d}:00 → Enero: {sol_enero} | Junio: {sol_junio}")

# ===============================================================
# PREPARAR FEATURES
# ===============================================================

df['es_finde'] = df['dia_semana'].isin([5, 6]).astype(int)

# Cíclicas
df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

# Interacciones
if 'Total Promedio de Radiación inclinada Solar R1(W/m²)' in df.columns:
    df['radiacion_x_sol'] = df['Total Promedio de Radiación inclinada Solar R1(W/m²)'] * df['tiene_sol']

feature_cols = [
    'hora', 'minuto', 'dia_semana', 'mes', 'dia_año', 'es_finde',
    'tiene_sol',  # ← NUEVA CARACTERÍSTICA CLAVE
    'hora_sin', 'hora_cos', 'dia_semana_sin', 'dia_semana_cos', 
    'mes_sin', 'mes_cos'
]

if 'Total Promedio de Radiación inclinada Solar R1(W/m²)' in df.columns:
    feature_cols.extend(['Total Promedio de Radiación inclinada Solar R1(W/m²)', 'radiacion_x_sol'])
if 'Total Promedio de Temperatura del módulo 1(°C)' in df.columns:
    feature_cols.append('Total Promedio de Temperatura del módulo 1(°C)')

# Lags
for lag in [1, 2, 6, 24]:
    df[f'lag_{lag}'] = df['generacion'].shift(lag)
    feature_cols.append(f'lag_{lag}')

for window in [6, 24]:
    df[f'ma_{window}'] = df['generacion'].rolling(window=window).mean()
    feature_cols.append(f'ma_{window}')

df_clean = df.dropna().reset_index(drop=True)

print(f"✅ Features: {len(feature_cols)} (incluye 'tiene_sol')")

# ===============================================================
# SPLIT 80/20
# ===============================================================

split_idx = int(len(df_clean) * 0.8)
df_train = df_clean.iloc[:split_idx].copy()
df_test = df_clean.iloc[split_idx:].copy()

X_train = df_train[feature_cols].values
y_train = df_train['generacion'].values
X_test = df_test[feature_cols].values
y_test = df_test['generacion'].values

print(f"\n✂️  Split: Train={len(df_train)} (80%) | Test={len(df_test)} (20%)")

# ===============================================================
# ENTRENAR
# ===============================================================

print("\n🚀 Entrenando modelo con restricción de horas de sol...")

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.9,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)
print("✅ Modelo entrenado")

# ===============================================================
# GUARDAR MODELO Y FEATURES
# ===============================================================
modelo_info = {
    'model': model,
    'feature_cols': feature_cols,
    'scaler_info': {
        'generacion_mean': df['generacion'].mean(),
        'generacion_std': df['generacion'].std(),
        'max_generacion': df['generacion'].max()
    }
}

with open('modelo_solar_entrenado.pkl', 'wb') as f:
    pickle.dump(modelo_info, f)
print("💾 Modelo guardado en 'modelo_solar_entrenado.pkl'")

# ===============================================================
# PREDICCIONES CON CORRECCIÓN
# ===============================================================

print("\n🔮 Generando predicciones con corrección por horas de sol...")

y_pred = np.maximum(0, model.predict(X_test))

# CORRECCIÓN CLAVE: Forzar 0 si no tiene sol
predicciones_corregidas = 0
for i, row in df_test.iterrows():
    hora = row['hora']
    mes = row['mes']
    
    # Si no tiene sol según el mes, forzar a 0
    if not hora_tiene_sol(hora, mes):
        if y_pred[i - df_test.index[0]] > 0:
            predicciones_corregidas += 1
        y_pred[i - df_test.index[0]] = 0

print(f"✅ Predicciones corregidas: {predicciones_corregidas} valores forzados a 0")

# ===============================================================
# MÉTRICAS
# ===============================================================

print("\n" + "=" * 80)
print("📊 MÉTRICAS CON CORRECCIÓN DE HORAS DE SOL")
print("=" * 80)

mae_global = mean_absolute_error(y_test, y_pred)
rmse_global = np.sqrt(mean_squared_error(y_test, y_pred))
r2_global = r2_score(y_test, y_pred)

print(f"\n📈 Métricas Globales:")
print(f"   MAE:  {mae_global:.2f} kWh")
print(f"   RMSE: {rmse_global:.2f} kWh")
print(f"   R²:   {r2_global:.5f}")

# Solo horas con sol
mask_sol = df_test['tiene_sol'] == 1
y_test_sol = y_test[mask_sol.values]
y_pred_sol = y_pred[mask_sol.values]

mae_sol = mean_absolute_error(y_test_sol, y_pred_sol)
rmse_sol = np.sqrt(mean_squared_error(y_test_sol, y_pred_sol))
r2_sol = r2_score(y_test_sol, y_pred_sol)
mape_sol = np.mean(np.abs((y_test_sol[y_test_sol > 0] - y_pred_sol[y_test_sol > 0]) / y_test_sol[y_test_sol > 0])) * 100

print(f"\n☀️  Métricas en HORAS CON SOL:")
print(f"   MAE:  {mae_sol:.2f} kWh")
print(f"   RMSE: {rmse_sol:.2f} kWh")
print(f"   R²:   {r2_sol:.5f}")
print(f"   MAPE: {mape_sol:.2f}%")

print("=" * 80)

# ===============================================================
# ANÁLISIS DEL PROBLEMA DE LAS 19:00
# ===============================================================

print("\n" + "=" * 80)
print("🔍 ANÁLISIS: ¿Se corrigió el problema de las 19:00?")
print("=" * 80)

df_test['y_pred'] = y_pred

# Análisis por hora en enero
df_enero_test = df_test[df_test['mes'] == 1]
if len(df_enero_test) > 0:
    print("\n📅 ENERO (mes del problema):")
    for hora in [17, 18, 19, 20]:
        df_hora = df_enero_test[df_enero_test['hora'] == hora]
        if len(df_hora) > 0:
            tiene_sol = hora_tiene_sol(hora, 1)
            pred_prom = df_hora['y_pred'].mean()
            real_prom = df_hora['generacion'].mean()
            estado = "✅" if (not tiene_sol and pred_prom == 0) or tiene_sol else "⚠️"
            sol_txt = "SÍ" if tiene_sol else "NO"
            print(f"   {estado} {hora}:00 - Sol:{sol_txt} | "
                  f"Real:{real_prom:6.1f} kWh | Pred:{pred_prom:6.1f} kWh")

# Comparar antes/después de la corrección
print("\n📊 Predicciones a las 19:00 por mes (DESPUÉS de corrección):")
for mes in range(1, 13):
    df_mes_19h = df_test[(df_test['mes'] == mes) & (df_test['hora'] == 19)]
    if len(df_mes_19h) > 0:
        pred_prom = df_mes_19h['y_pred'].mean()
        real_prom = df_mes_19h['generacion'].mean()
        tiene_sol = hora_tiene_sol(19, mes)
        mes_nombre = ['', 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                      'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][mes]
        
        if tiene_sol:
            estado = "✅"
            nota = "(correcto, tiene sol)"
        else:
            if pred_prom == 0:
                estado = "✅"
                nota = "(CORREGIDO: pred=0)"
            else:
                estado = "⚠️"
                nota = "(¿aún tiene valores?)"
        
        print(f"   {estado} {mes_nombre}: Real:{real_prom:6.1f} | Pred:{pred_prom:6.1f} {nota}")

# ===============================================================
# GRÁFICAS
# ===============================================================

print("\n📊 Generando gráficas...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Gráfica 1: Superposición
ax1 = axes[0, 0]
n = min(300, len(df_test))
ax1.plot(df_test['fecha_y_hora'].iloc[:n], y_test[:n], 'b-', label='Real', linewidth=2)
ax1.plot(df_test['fecha_y_hora'].iloc[:n], y_pred[:n], 'r-', label='Predicción Corregida', linewidth=2)
ax1.set_title('Superposición con Corrección de Horas de Sol', fontsize=14, fontweight='bold')
ax1.set_xlabel('Fecha y Hora')
ax1.set_ylabel('Generación (kWh)')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Gráfica 2: Patrón diario por mes
ax2 = axes[0, 1]
if len(df_enero_test) > 0:
    patron_real = df_enero_test.groupby('hora')['generacion'].mean()
    patron_pred = df_enero_test.groupby('hora')['y_pred'].mean()
    ax2.plot(patron_real.index, patron_real.values, 'b-o', label='Real', linewidth=3, markersize=8)
    ax2.plot(patron_pred.index, patron_pred.values, 'r-s', label='Predicción', linewidth=3, markersize=8)
    ax2.axvline(x=18, color='orange', linewidth=2, linestyle='--', label='Puesta sol (18h)')
    ax2.axvline(x=19, color='red', linewidth=2, linestyle='--', alpha=0.5, label='19h (antes problema)')
    ax2.set_title('Patrón Diario en ENERO\n(Mes del problema - CORREGIDO)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hora del día')
    ax2.set_ylabel('Generación promedio (kWh)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))

# Gráfica 3: Horas de sol por mes
ax3 = axes[1, 0]
meses = range(1, 13)
duraciones = [HORAS_SOL_MADRID[m]['puesta'] - HORAS_SOL_MADRID[m]['salida'] for m in meses]
ax3.bar(meses, duraciones, color='orange', alpha=0.7, edgecolor='black')
for m in meses:
    ax3.text(m, duraciones[m-1] + 0.2, f"{duraciones[m-1]:.1f}h", 
             ha='center', fontsize=9, fontweight='bold')
ax3.set_title('Duración de Luz Solar por Mes\n(Madrid)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Mes')
ax3.set_ylabel('Horas de luz')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(range(1, 13))

# Gráfica 4: Importancia de features
ax4 = axes[1, 1]
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:15]
features_top = [feature_cols[i] for i in indices]
importances_top = importances[indices]
colores = ['green' if 'sol' in f else 'blue' for f in features_top]
ax4.barh(range(len(features_top)), importances_top, color=colores, alpha=0.7)
ax4.set_yticks(range(len(features_top)))
ax4.set_yticklabels(features_top, fontsize=9)
ax4.set_xlabel('Importancia')
ax4.set_title('Top 15 Características\n(Verde = relacionadas con sol)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('MODELO_CORREGIDO_HORAS_SOL.png', dpi=200, bbox_inches='tight')
print("✅ Gráfica guardada: MODELO_CORREGIDO_HORAS_SOL.png")

# ===============================================================
# GUARDAR RESULTADOS
# ===============================================================

resultados = df_test[['fecha_y_hora', 'hora', 'mes', 'tiene_sol', 'generacion']].copy()
resultados['Prediccion_Corregida'] = y_pred
resultados['Error'] = y_test - y_pred
resultados.to_excel('RESULTADOS_CORREGIDOS_SOL.xlsx', index=False)

print("✅ Resultados guardados: RESULTADOS_CORREGIDOS_SOL.xlsx")

# ===============================================================
# RESUMEN
# ===============================================================

print("\n" + "=" * 80)
print("🎉 CORRECCIÓN COMPLETADA")
print("=" * 80)

print("\n🌅 HORAS DE SOL APLICADAS (Madrid):")
for mes in range(1, 13):
    mes_nombre = ['', 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                  'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][mes]
    duracion = HORAS_SOL_MADRID[mes]['puesta'] - HORAS_SOL_MADRID[mes]['salida']
    print(f"   {mes_nombre}: {HORAS_SOL_MADRID[mes]['salida']:.1f}h - {HORAS_SOL_MADRID[mes]['puesta']:.1f}h ({duracion:.1f}h luz)")

print(f"\n📊 RESULTADOS:")
print(f"   R² en horas con sol: {r2_sol:.5f}")
print(f"   MAE: {mae_sol:.2f} kWh")
print(f"   MAPE: {mape_sol:.2f}%")
print(f"   Predicciones corregidas: {predicciones_corregidas}")

print("\n✅ El problema de las 19:00 ha sido CORREGIDO")
print("   Ahora las predicciones respetan las horas de luz reales de España")

print("=" * 80)

plt.show()

