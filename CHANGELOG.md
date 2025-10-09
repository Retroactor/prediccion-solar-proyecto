# Changelog

Todos los cambios notables de este proyecto serán documentados en este archivo.

## [2.0.0] - 2025-01-09

### ✨ Añadido
- **Horas de sol reales de España** por mes (Madrid como referencia)
- **Corrección automática** de predicciones según horario solar
- **Característica `tiene_sol`** que varía por mes
- **Información interactiva** en sidebar con horas de sol
- **Predicciones futuras corregidas** (1-30 días)
- **Análisis por día específico** con detalles horarios

### 🔧 Corregido
- **Picos extraños a las 19:00** en meses de invierno (ahora pred=0 cuando no hay sol)
- **Error de datetime** en predicciones futuras (`AttributeError`)
- **Predicciones nocturnas** ahora correctamente en 0
- **Ajuste temporal** según zona horaria europea

### 🎨 Mejorado
- **Dashboard interactivo** más intuitivo con 3 modos de exploración
- **Gráficas de superposición** con zoom y hover
- **Métricas en tiempo real** según período seleccionado
- **Documentación completa** con ejemplos de uso

### 📊 Métricas Finales
- R² = 0.953 (en horas solares)
- MAE = 1,194 kWh
- MAPE = 13.35%
- 364 predicciones corregidas automáticamente

---

## [1.0.0] - 2025-01-08

### ✨ Versión Inicial
- Dashboard básico con análisis temporal
- Modelo Random Forest para predicciones
- Split 80/20 para validación
- Gráficas estáticas de resultados
- Análisis por minutos y horas
- Detección básica de anomalías

### ⚠️ Problemas Conocidos
- Predicciones incorrectas en horas sin sol
- No considera variación estacional de luz
- Dashboard menos interactivo

---

## Formato

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

### Tipos de Cambios
- **✨ Añadido** - Para nuevas características
- **🔧 Corregido** - Para corrección de bugs
- **🎨 Mejorado** - Para mejoras en código existente
- **🗑️ Eliminado** - Para características eliminadas
- **⚠️ Deprecado** - Para características que serán eliminadas próximamente
- **🔒 Seguridad** - Para cambios de seguridad

