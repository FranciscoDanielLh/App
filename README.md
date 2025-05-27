# 📊 Plataforma de Análisis - Proyecto Final

**Aplicación web interactiva para Media Mix Modeling (MMM) y Segmentación de Clientes**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://apptsmcdfinal.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 **Descripción del Proyecto**

Esta aplicación web desarrollada en **Streamlit** implementa dos módulos principales de análisis de marketing y clientes:

1. **📈 Media Mix Modeling (MMM)** - Modelado de atribución publicitaria con transformaciones de adstock y análisis de ROI
2. **👥 Segmentación de Clientes** - Clustering avanzado con análisis demográfico y exportación de resultados

**Desarrollado como proyecto final** para cursos de Marketing Analytics, Data Science o similares.

---

## ✨ **Características Principales**

### **📈 Módulo MMM (Media Mix Modeling)**
- ✅ **Carga de datos CSV** con validación automática
- ✅ **Análisis exploratorio** con series temporales y correlaciones
- ✅ **Configuración flexible** de variables objetivo y medios
- ✅ **Parámetros de adstock** configurables por medio
- ✅ **Modelado DLM** (Dynamic Linear Model) con estacionalidad Fourier
- ✅ **Visualizaciones interactivas**: gráficos apilados, pie charts, ROI
- ✅ **Exportación a Excel** y guardado de modelos
- ✅ **Interfaz intuitiva** con flujo guiado paso a paso

### **👥 Módulo Segmentación**
- ✅ **Clustering con K-means** y análisis del método del codo
- ✅ **Métricas de validación**: AIC, BIC, Silhouette Score
- ✅ **Análisis de relevancia** de variables por cluster
- ✅ **Integración de variables demográficas** para análisis cruzado
- ✅ **Visualizaciones avanzadas**: boxplots, distribuciones, importancia
- ✅ **Exportación completa**: tablas cruzadas y asignaciones
- ✅ **Detección automática** del número óptimo de clusters

---

## 🚀 **Instalación y Ejecución**

### **Requisitos Previos**
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### **1. Clonar o Descargar el Proyecto**
```bash
# Si usas Git
git clone <tu-repositorio>
cd tu-proyecto

# O simplemente descarga los archivos del proyecto
```

### **2. Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### **3. Ejecutar la Aplicación**
```bash
streamlit run app.py
```

### **4. Abrir en el Navegador**
La aplicación se abrirá automáticamente en `http://localhost:8501`

---

## 📁 **Estructura del Proyecto**

```
proyecto-final/
├── app.py                     # 🔧 Aplicación principal
├── requirements.txt           # 📦 Dependencias del proyecto
├── README.md                 # 📖 Este archivo
├── datos_muestra/            # 📊 Datos de ejemplo
│   ├── mmm_sample.csv        # Datos para MMM
│   ├── segmentacion.csv      # Datos para segmentación
│   └── demografia.csv        # Variables demográficas
└── exports/                  # 💾 Archivos exportados (se crea automáticamente)
```

---

## 📊 **Uso de la Aplicación**

### **🎯 Navegación Principal**
Usa el **sidebar izquierdo** para alternar entre:
- **📈 MMM (Media Mix Modeling)**
- **👥 Segmentación de Clientes**

---

## 📈 **Módulo MMM - Guía de Uso**

### **Pestaña 1: Input y Configuración**

#### **1. Cargar Datos**
- Sube un archivo **CSV** con las siguientes columnas obligatorias:
  - `Fecha` - Período temporal
  - `Ventas_Unidades` o `Ventas_Revenue` - Variable objetivo
  - `Inversion_[Medio]` - Inversión por medio publicitario
  - `Impresiones_[Medio]` - Actividades por medio

#### **2. Análisis Exploratorio**
- **Series temporales** automáticas de todas las variables
- **Estadísticas de correlación** generales
- **Métricas descriptivas** del dataset

#### **Estructura de Datos Esperada:**
```csv
Fecha,Ventas_Revenue,Inversion_TV,Inversion_Digital,Impresiones_TV,Impresiones_Digital
2024-01-01,98000,25000,15000,1250000,750000
2024-01-08,95200,24000,14500,1200000,725000
...
```

### **Pestaña 2: Modelado y Resultados**

#### **1. Configuración de Variables**
- **Variable Objetivo**: Selecciona la métrica de ventas a modelar
- **Variables de Medios**: Elige qué medios incluir en el análisis
- **Frecuencia**: Semanal o mensual

#### **2. Parámetros del Modelo DLM**
- **Discount Factor Base**: Factor de descuento para base orgánica (0-1)
- **Punto Inicial Base**: Valor inicial para base orgánica
- **Estacionalidad Fourier**: Incluir componente estacional

#### **3. Parámetros de Adstock**
- **Configuración individual** por medio (0-1)
- **0**: Sin carry-over effect
- **1**: Máximo carry-over effect

#### **4. Ejecución y Resultados**
- **Gráfico apilado** de contribuciones por medio
- **Pie chart** con ROI por medio
- **Tabla detallada** de ROI
- **Descarga a Excel** con todos los resultados
- **Guardar modelo** para uso posterior

---

## 👥 **Módulo Segmentación - Guía de Uso**

### **Pestaña 1: Carga de Datos**

#### **1. Cargar Dataset Principal**
- Sube archivo **CSV** con variables numéricas y continuas
- Selecciona **variables para clustering**
- Revisa **estadísticas descriptivas**

#### **Estructura de Datos Esperada:**
```csv
Respondent_ID,Edad,Ingresos,Gasto_Mensual,Frecuencia_Compra,Satisfaccion
1,25,45000,320,8,4.2
2,34,67000,480,12,4.8
...
```

### **Pestaña 2: Clustering**

#### **1. Configuración**
- **Rango de clusters**: Define mínimo y máximo
- **Ejecutar análisis**: K-means con múltiples métricas

#### **2. Resultados Automáticos**
- **Gráfico del codo** con 4 métricas (Inercia, Silhouette, AIC, BIC)
- **Número óptimo** sugerido automáticamente
- **Distribución de clusters** visualizada
- **Tabla de métricas** por número de clusters

### **Pestaña 3: Probabilidades**

#### **1. Análisis Detallado**
- **Centros de clusters** con valores promedio
- **Boxplots interactivos** por variable y cluster
- **Relevancia de variables** para distinguir clusters
- **Gráfico de importancia** normalizada

### **Pestaña 4: Demografía**

#### **1. Variables Adicionales**
- **Cargar CSV demográfico** para enriquecer análisis
- **Merge automático** por Respondent_ID

#### **2. Análisis Cruzado**
- **Resumen por cluster** con métricas básicas
- **Variables distintivas** automáticas
- **Estadísticas completas** expandibles

#### **3. Exportación**
- **📥 Tablas Cruzadas**: Excel con múltiples hojas
- **📋 Solo Asignaciones**: Respondent_ID + Cluster

---

## 🛠 **Tecnologías Utilizadas**

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Streamlit** | 1.28.1 | Framework de aplicación web |
| **Pandas** | 2.1.3 | Manipulación y análisis de datos |
| **NumPy** | 1.24.3 | Cálculos numéricos |
| **Plotly** | 6.1.1 | Visualizaciones interactivas |
| **Scikit-learn** | 1.3.2 | Algoritmos de machine learning |
| **SciPy** | 1.11.4 | Estadísticas avanzadas |
| **OpenPyXL** | 3.1.2 | Exportación a Excel |

---


## 📄 **Licencia**

Este proyecto se distribuye bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.
