# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto de Predicción Solar España!

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [Cómo Contribuir](#cómo-contribuir)
- [Reportar Bugs](#reportar-bugs)
- [Sugerir Mejoras](#sugerir-mejoras)
- [Pull Requests](#pull-requests)
- [Estilo de Código](#estilo-de-código)

## 📜 Código de Conducta

Este proyecto se adhiere a un código de conducta. Al participar, se espera que respetes este código.

## 🚀 Cómo Contribuir

### 1. Fork el Proyecto

```bash
# Fork en GitHub, luego:
git clone https://github.com/tu-usuario/prediccion-solar-espana.git
cd prediccion-solar-espana
```

### 2. Crea una Rama

```bash
git checkout -b feature/mi-nueva-caracteristica
```

**Nombres de ramas sugeridos:**
- `feature/nombre` - Para nuevas características
- `fix/descripcion` - Para corrección de bugs
- `docs/actualizacion` - Para documentación
- `refactor/componente` - Para refactorización

### 3. Realiza tus Cambios

- Escribe código claro y comentado
- Añade docstrings a funciones
- Actualiza documentación si es necesario
- Prueba tus cambios

### 4. Commit

```bash
git add .
git commit -m "Añadir: descripción clara del cambio"
```

**Formato de commits:**
- `Añadir:` - Nuevas características
- `Corregir:` - Bugs
- `Actualizar:` - Documentación
- `Refactorizar:` - Mejoras de código

### 5. Push y Pull Request

```bash
git push origin feature/mi-nueva-caracteristica
```

Luego crea un Pull Request en GitHub con:
- Título descriptivo
- Descripción de cambios
- Screenshots si aplica
- Referencias a issues relacionados

## 🐛 Reportar Bugs

Al reportar un bug, incluye:

### Información Esencial
- **Descripción clara** del problema
- **Pasos para reproducir**
- **Comportamiento esperado** vs **actual**
- **Screenshots** si es visual
- **Versión de Python** y dependencias
- **Sistema operativo**

### Template de Bug Report

```markdown
## Descripción
[Descripción clara y concisa del bug]

## Pasos para Reproducir
1. Ir a '...'
2. Hacer clic en '...'
3. Ver error

## Comportamiento Esperado
[Qué debería pasar]

## Comportamiento Actual
[Qué pasa realmente]

## Screenshots
[Si aplica]

## Entorno
- OS: [e.g. Windows 10]
- Python: [e.g. 3.9.0]
- Streamlit: [e.g. 1.28.0]
```

## 💡 Sugerir Mejoras

### Antes de Sugerir
- Revisa si ya existe un issue similar
- Verifica la documentación actual
- Asegúrate de que la mejora tiene sentido

### Template de Mejora

```markdown
## Problema que Resuelve
[Descripción del problema actual]

## Solución Propuesta
[Descripción de tu solución]

## Alternativas Consideradas
[Otras opciones que consideraste]

## Contexto Adicional
[Cualquier otra información relevante]
```

## 🔄 Pull Requests

### Checklist Antes de Enviar

- [ ] El código sigue el estilo del proyecto
- [ ] He comentado mi código en secciones complejas
- [ ] He actualizado la documentación
- [ ] Mis cambios no generan nuevas advertencias
- [ ] He probado localmente
- [ ] He actualizado CHANGELOG.md

### Revisión de Código

Tu PR será revisado por:
1. **Corrección** - ¿Funciona?
2. **Estilo** - ¿Sigue las convenciones?
3. **Documentación** - ¿Está documentado?
4. **Pruebas** - ¿Funciona en diferentes escenarios?

## 🎨 Estilo de Código

### Python

```python
# Buenos nombres de variables
horas_sol_madrid = {...}
modelo_gradient_boosting = GradientBoostingRegressor()

# Funciones documentadas
def hora_tiene_sol(hora, mes):
    """
    Determinar si una hora tiene sol según el mes.
    
    Args:
        hora (int): Hora del día (0-23)
        mes (int): Mes del año (1-12)
    
    Returns:
        bool: True si hay sol, False si no
    """
    pass

# Imports organizados
# 1. Estándar
import sys
from datetime import datetime

# 2. Terceros
import pandas as pd
import numpy as np

# 3. Locales
from utils import procesar_datos
```

### Convenciones

- **Nombres**: snake_case para variables y funciones
- **Constantes**: MAYUSCULAS_CON_GUION_BAJO
- **Clases**: PascalCase
- **Líneas**: Máximo 88 caracteres (Black formatter)
- **Docstrings**: Google style
- **Comentarios**: En español, claros y concisos

### Estructura de Archivos

```python
# 1. Docstring del módulo
"""
Descripción del módulo
"""

# 2. Imports

# 3. Constantes globales

# 4. Funciones auxiliares

# 5. Función principal

# 6. if __name__ == "__main__"
```

## 🧪 Testing

Aunque actualmente no hay tests automatizados, al contribuir:

1. **Prueba manualmente** tus cambios
2. **Verifica diferentes escenarios**:
   - Enero vs Junio (horas de sol)
   - Días con/sin datos
   - Rangos de fechas variados
3. **Verifica el dashboard** funciona correctamente

## 📚 Documentación

Al añadir código:

- Actualiza **README.md** si cambias funcionalidad
- Actualiza **CHANGELOG.md** con tus cambios
- Comenta código complejo
- Añade docstrings a nuevas funciones

## 🎯 Áreas de Contribución

### Fácil 🟢
- Corrección de typos
- Mejoras de documentación
- Añadir comentarios
- Mejorar mensajes de error

### Medio 🟡
- Nuevas visualizaciones
- Optimización de rendimiento
- Mejoras de UI/UX
- Refactorización de código

### Difícil 🔴
- Nuevos modelos de ML
- Cambios arquitectónicos
- Optimizaciones complejas
- Nuevas características mayores

## 🙏 Reconocimientos

Todos los contribuidores serán añadidos al README.md en la sección de Contribuidores.

## ❓ Preguntas

Si tienes dudas:
1. Revisa la documentación
2. Busca en issues existentes
3. Crea un nuevo issue con tag `question`

---

**¡Gracias por contribuir!** 🎉

