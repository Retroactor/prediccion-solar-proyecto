# 🔋 Predicción de Generación Solar - España

Dashboard interactivo para análisis y predicción de generación de energía solar, optimizado para condiciones reales de España.

## 📊 Características Principales

### ✅ Dashboard Interactivo
- **Análisis histórico** con visualizaciones de superposición Real vs Predicción
- **Predicciones futuras** de 1 a 30 días
- **Exploración por día** con análisis detallado
- **Gráficas interactivas** con zoom, hover y exportación

### ✅ Modelo Optimizado
- **Gradient Boosting Regressor** con R² > 0.95
- **Ajustado para España**: respeta horas de sol reales por mes
- **Split 80/20** para validación robusta
- **Características avanzadas**: cíclicas, lags, medias móviles

### ✅ Correcciones Específicas para España
- **Horas de sol variables por mes** (9.5h en invierno, 14.5h en verano)
- **Sin predicciones imposibles** (no predice generación de noche)
- **Zona horaria europea** considerada
- **Patrones estacionales** capturados correctamente

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/prediccion-solar-espana.git
cd prediccion-solar-espana

# Instalar dependencias
pip install -r requirements_final.txt
```

### Datos

⚠️ **Los datos no están incluidos en el repositorio** por privacidad.

Para usar el proyecto necesitas:
1. Obtener un archivo Excel con datos de generación solar
2. Guardarlo como `Datos reales.xlsx` en el directorio raíz
3. El archivo debe tener las columnas: `fecha`, `fecha y hora`, `generacion`

### Ejecutar Dashboard

```bash
streamlit run dashboard_interactivo.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`

### Entrenar Modelo

```bash
python modelo_corregido_horas_sol.py
```

Esto generará:
- `MODELO_CORREGIDO_HORAS_SOL.png` - Gráficas de validación
- `RESULTADOS_CORREGIDOS_SOL.xlsx` - Predicciones detalladas

## 📁 Estructura del Proyecto

```
ProyectoElectrico/
├── dashboard_interactivo.py          # Dashboard principal (Streamlit)
├── modelo_corregido_horas_sol.py    # Script de entrenamiento
├── MODELO_CORREGIDO_HORAS_SOL.png   # Visualizaciones de ejemplo
├── requirements_final.txt            # Dependencias Python
├── README.md                         # Este archivo
├── LICENSE                           # Licencia MIT
├── CHANGELOG.md                      # Historial de cambios
└── CONTRIBUTING.md                   # Guía de contribución

# Archivos locales (no incluidos en Git)
├── Datos reales.xlsx                 # Tus datos (añádelos localmente)
└── RESULTADOS_CORREGIDOS_SOL.xlsx   # Resultados generados
```

## 🎯 Uso del Dashboard

### Modo 1: Datos Históricos
1. Selecciona rango de fechas
2. Visualiza 4 gráficas principales:
   - Superposición temporal
   - Zoom detallado
   - Scatter de correlación
   - Patrón diario promedio
3. Analiza métricas (MAE, R², MAPE)

### Modo 2: Predicciones Futuras
1. Elige días hacia adelante (1-30)
2. Observa predicciones
3. Descarga resultados en CSV
4. **Nota**: Las predicciones respetan horas de sol del mes

### Modo 3: Día Específico
1. Selecciona una fecha
2. Ve análisis hora por hora
3. Compara real vs predicción
4. Revisa tabla detallada

## 📊 Métricas del Modelo

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R²** | 0.953 | Excelente ajuste |
| **MAE** | 1,194 kWh | Error promedio aceptable |
| **MAPE** | 13.35% | Precisión muy buena |

## 🌅 Horas de Sol en España (Madrid)

El modelo considera las horas de sol reales por mes:

| Mes | Horas de Luz | Salida-Puesta |
|-----|--------------|---------------|
| Enero | 9.5h | 8:30 - 18:00 |
| Junio | 14.5h | 7:00 - 21:30 |
| Diciembre | 9.5h | 8:30 - 18:00 |

**Importancia**: Evita predicciones imposibles (ej: generación a las 19:00 en invierno)

## 🔧 Personalización

### Ajustar para tu ubicación

Si no estás en Madrid, edita `dashboard_interactivo.py`:

```python
HORAS_SOL_MADRID = {
    1: {'salida': 8.5, 'puesta': 18.0},  # Ajustar aquí
    # ... resto de meses
}
```

**Guía**:
- Barcelona: -30min aprox
- Galicia: +1h aprox
- Andalucía: similar a Madrid

## 📈 Características del Modelo

### Features Principales
- **Temporales**: hora, día, mes, día del año
- **Cíclicas**: transformaciones seno/coseno para periodicidad
- **Solares**: intensidad solar, horas de sol
- **Físicas**: radiación, temperatura de módulos
- **Históricas**: lags, medias móviles
- **España-específicas**: `tiene_sol` por mes

### Algoritmo
- **Gradient Boosting Regressor**
  - 200 estimadores
  - Profundidad máxima: 12
  - Learning rate: 0.05
  - Subsample: 0.9

## 🛠️ Tecnologías

- **Python 3.8+**
- **Streamlit** - Dashboard interactivo
- **Scikit-learn** - Machine Learning
- **Pandas** - Manipulación de datos
- **Plotly** - Visualizaciones interactivas
- **NumPy** - Cálculos numéricos

## 📝 Dependencias

Ver `requirements_final.txt` para lista completa. Principales:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
```

## 🎓 Metodología

1. **Carga de datos**: Excel con generación solar histórica
2. **Feature engineering**: 35+ características creadas
3. **Split temporal**: 80% train, 20% test
4. **Entrenamiento**: Gradient Boosting optimizado
5. **Validación**: Métricas en horas solares
6. **Corrección**: Ajuste según horas de sol reales
7. **Visualización**: Dashboard interactivo

## 🔍 Casos de Uso

### 1. Planificación de Mantenimiento
```
Modo: Predicciones Futuras
Días: 14
Acción: Identificar días de baja generación
```

### 2. Análisis de Rendimiento
```
Modo: Datos Históricos
Período: Último mes
Acción: Comparar real vs esperado
```

### 3. Detección de Anomalías
```
Modo: Día Específico
Acción: Buscar días con gran error
Diagnóstico: Posible fallo de sistema
```

## 🌟 Ventajas del Proyecto

- ✅ **Adaptado a España**: Horas de sol reales
- ✅ **Alta precisión**: R² > 0.95
- ✅ **Fácil de usar**: Dashboard intuitivo
- ✅ **Interactivo**: Exploración visual
- ✅ **Exportable**: Resultados en CSV/Excel
- ✅ **Documentado**: Código claro y comentado

## 🐛 Troubleshooting

### Dashboard no se abre
```bash
# Verificar puerto
streamlit run dashboard_interactivo.py --server.port 8502
```

### Error al cargar datos
- Verificar que `Datos reales.xlsx` está en el directorio
- Revisar formato de fechas en Excel

### Predicciones extrañas
- Verificar que las horas de sol corresponden a tu ubicación
- Ajustar `HORAS_SOL_MADRID` si es necesario

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.

## 👤 Autor

Desarrollado con ❤️ considerando las condiciones solares reales de España 🇪🇸

## 🙏 Agradecimientos

- Datos de generación solar reales
- Comunidad de Streamlit
- Scikit-learn por las herramientas de ML

---

**⭐ Si este proyecto te es útil, dale una estrella en GitHub!**

**Última actualización**: Versión 2.0 - Con corrección de horas de sol

