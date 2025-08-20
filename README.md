# TelecomX - Parte 2: Predicción de Churn de Clientes

## 📋 Propósito del Análisis

Este proyecto tiene como **objetivo principal** desarrollar modelos predictivos para identificar clientes con alta probabilidad de cancelación (churn) en la empresa de telecomunicaciones TelecomX. Mediante técnicas de machine learning, se busca:

- Predecir qué clientes tienen mayor riesgo de cancelar sus servicios
- Identificar los factores más influyentes en la decisión de cancelación
- Proporcionar insights accionables para estrategias de retención de clientes
- Desarrollar un sistema de alerta temprana para intervenciones proactivas

## 📁 Estructura del Proyecto

```
telecomx_2/
│
├── README.md                           # Documentación del proyecto
├── requirements.txt                    # Dependencias del proyecto
│
├── data/
│   └── telecomX_processed.csv         # Dataset preprocesado y listo para análisis
│
├── notebooks/
│   └── telecomX_parte2.ipynb          # Notebook principal con análisis completo
│
└── secondpart/                        # Entorno virtual Python
    ├── Scripts/
    ├── Lib/
    └── pyvenv.cfg
```

### Descripción de Archivos Principales

- **`telecomX_parte2.ipynb`**: Notebook principal que contiene todo el flujo de trabajo del proyecto
- **`telecomX_processed.csv`**: Dataset con 7,043 registros de clientes ya preprocesados
- **`requirements.txt`**: Lista de librerías necesarias para ejecutar el proyecto

## 🔧 Proceso de Preparación de Datos

### 1. Clasificación de Variables

**Variables Categóricas:**
- `Churn` (variable objetivo)
- `gender`, `MultipleLines`, `InternetService`
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`
- `TechSupport`, `StreamingTV`, `StreamingMovies`
- `Contract`, `PaymentMethod`

**Variables Numéricas:**
- `tenure` (meses de permanencia)
- `Charges.Monthly` (cargo mensual)
- `Charges.Total` (cargo total acumulado)
- `Cuentas_Diarias` (cuentas por día)

### 2. Etapas de Transformación

#### **Codificación de Variables Categóricas**
```python
# Aplicación de One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
```

#### **Balanceado de Clases**
```python
# Uso de SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

#### **Normalización de Variables Numéricas**
```python
# StandardScaler para variables con diferentes escalas
scaler = StandardScaler()
df_scaled[columns_to_scale] = scaler.fit_transform(df_balanced[columns_to_scale])
```

### 3. Separación de Datos

- **División**: 80% entrenamiento / 20% prueba
- **Estratificación**: Mantiene proporción de churn en ambos conjuntos
- **Semilla aleatoria**: 42 (para reproducibilidad)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## 🤖 Modelos Implementados

### Algoritmos Utilizados

1. **Regresión Logística** (con normalización completa)
2. **K-Nearest Neighbors (KNN)** (con normalización completa)
3. **Árbol de Decisión** (sin normalización adicional)
4. **Random Forest** (sin normalización adicional)

### Justificaciones Metodológicas

**Normalización Diferenciada:**
- **Modelos sensibles a escala** (Regresión Logística, KNN): Requieren normalización completa
- **Modelos basados en árboles** (Decision Tree, Random Forest): Robustos a diferencias de escala

**Métricas de Evaluación:**
- **Accuracy**: Rendimiento general
- **Precision**: Calidad de predicciones positivas
- **Recall**: Capacidad de detectar churn (crítico para el negocio)
- **F1-Score**: Balance entre precision y recall
- **ROC-AUC**: Capacidad de discriminación

## 📊 Insights del Análisis Exploratorio

### Principales Hallazgos

1. **Tiempo de Permanencia**: Clientes con menos de 12 meses tienen mayor riesgo de churn
2. **Tipo de Contrato**: Contratos mes a mes muestran mayor cancelación
3. **Método de Pago**: Cheque electrónico correlaciona fuertemente con churn
4. **Servicios Adicionales**: Clientes sin servicios adicionales tienden a cancelar más

### Visualizaciones Clave

- **Distribución de churn por tiempo de permanencia**
- **Impacto del tipo de contrato en la retención**
- **Correlación entre gasto total y probabilidad de churn**
- **Matrices de confusión para cada modelo**

## 🏆 Resultados Principales

### Rendimiento de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|---------|----------|---------|
| **Random Forest** | 0.8234 | 0.8053 | 0.8525 | 0.8282 | 0.8234 |
| Árbol de Decisión | 0.8047 | 0.7744 | 0.8574 | 0.8139 | 0.8047 |
| Regresión Logística | 0.7918 | 0.7740 | 0.8197 | 0.7963 | 0.7918 |
| KNN | 0.7560 | 0.7317 | 0.8033 | 0.7659 | 0.7560 |

### Variables Más Importantes

1. **tenure** (tiempo de permanencia)
2. **Charges.Total** (gasto total acumulado)
3. **Contract_Two year** (contrato de dos años)
4. **InternetService_Fiber optic** (servicio de fibra óptica)
5. **PaymentMethod_Electronic check** (pago con cheque electrónico)

## 🚀 Instrucciones de Ejecución

### 1. Requisitos del Sistema

- **Python**: 3.8 o superior
- **Jupyter Notebook** o **JupyterLab**
- **Git** (para clonar el repositorio)

### 2. Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/telecomx_2.git
cd telecomx_2

# Crear entorno virtual (recomendado)
python -m venv telecom_env
telecom_env\Scripts\activate  # Windows
# source telecom_env/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Librerías Requeridas

```
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
imbalanced-learn>=0.10.0
jupyter>=1.0.0
```

### 4. Ejecución del Proyecto

```bash
# Activar entorno virtual
telecom_env\Scripts\activate

# Iniciar Jupyter Notebook
jupyter notebook

# Abrir el archivo: notebooks/telecomX_parte2.ipynb
```

### 5. Orden de Ejecución

1. **Preparación de datos**: Celdas 1-10
2. **Análisis exploratorio**: Celdas 11-15
3. **Separación de datos**: Celdas 16-20
4. **Creación de modelos**: Celdas 21-30
5. **Evaluación**: Celdas 31-40
6. **Interpretación**: Celdas 41-50

## 💡 Recomendaciones de Uso

### Para Estudiantes
- Ejecutar el notebook celda por celda para entender cada paso
- Experimentar con diferentes parámetros en los modelos
- Analizar las visualizaciones para extraer insights adicionales

### Para Implementación en Negocio
- Utilizar el modelo Random Forest para predicciones en producción
- Implementar sistema de monitoreo basado en las variables más importantes
- Desarrollar estrategias de retención basadas en los factores identificados

## 📈 Próximos Pasos

1. **Optimización de hiperparámetros** para mejorar rendimiento
2. **Implementación en tiempo real** del sistema predictivo
3. **Desarrollo de dashboard** interactivo para el área de negocio
4. **Evaluación continua** del modelo con nuevos datos

---

**Autor**: Samuel Mejia - Estudiante de Especialización en Ciencia de Datos  
**Fecha**: Agosto 2025  
**Contacto**: mejiabsamuel777@gmail.com
