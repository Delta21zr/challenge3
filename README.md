Proyecto de Predicción de Cancelación de Clientes (Churn)
Descripción

Este proyecto tiene como objetivo predecir la cancelación de clientes (churn) en una compañía de telecomunicaciones utilizando técnicas de machine learning. Se realiza un preprocesamiento completo de los datos, se entrenan distintos modelos de clasificación y se analiza la importancia de las variables para la predicción.

El flujo del proyecto incluye:

Carga y limpieza de datos.

Transformación de variables categóricas a formato numérico.

Análisis exploratorio y evaluación de desbalance de clases.

Normalización de datos para modelos sensibles a escala.

Entrenamiento y evaluación de modelos de machine learning.

Análisis de importancia de variables y visualización de correlaciones.

Estructura del proyecto
challenge1/
│
├─ challenge3/
│   ├─ challange3.py            # Código principal del proyecto
│
├─ challange2/
│   ├─ TelecomX_Datos_Limpios.csv  # CSV limpio y estandarizado
│
├─ .venv/                       # Entorno virtual de Python
│
└─ README.md                     # Este archivo

Requisitos

Python 3.9+

Bibliotecas de Python:

pip install pandas scikit-learn matplotlib seaborn


Entorno virtual recomendado (venv) para gestionar dependencias.

Descripción de los pasos realizados
1️⃣ Carga de datos

Se cargó un CSV previamente limpiado y estandarizado (TelecomX_Datos_Limpios.csv) que contiene únicamente las columnas relevantes para el análisis.

2️⃣ Preprocesamiento

Se eliminaron columnas que no aportan valor predictivo, como customerID.

Se transformaron las variables categóricas mediante one-hot encoding.

Se creó la variable objetivo Churn_Binario para indicar si un cliente canceló (1) o no (0).

3️⃣ Análisis exploratorio

Se evaluó la proporción de clientes que cancelaron y los que permanecieron activos.

Se visualizó la matriz de correlación de variables numéricas.

Se exploraron relaciones entre variables importantes y la cancelación mediante boxplots.

4️⃣ Normalización

Se aplicó StandardScaler a las variables numéricas para modelos sensibles a la escala, como Regresión Logística y KNN.

5️⃣ División de datos

Se dividió el dataset en:

70% entrenamiento

30% prueba

6️⃣ Entrenamiento de modelos

Se entrenaron tres modelos de clasificación:

Regresión Logística (sensible a la escala)

K-Nearest Neighbors (KNN) (sensible a la escala)

Random Forest (no sensible a la escala)

7️⃣ Evaluación de modelos

Se evaluaron los modelos con métricas clave:

Exactitud (Accuracy)

Precisión (Precision)

Recall

F1-score

Matriz de confusión

8️⃣ Análisis de importancia de variables

Regresión Logística: análisis de coeficientes para identificar variables con mayor contribución a la predicción.

Random Forest: análisis de importancia de variables basado en la reducción de impureza.

Resultados

Identificación de variables más relevantes para la predicción del churn.

Comparación de desempeño entre modelos sensibles y no sensibles a la escala.

Visualización de patrones en variables numéricas clave (tenure, MonthlyCharges, TotalCharges) frente al churn.

Uso

Activar el entorno virtual:

& "C:/Users/Alanl/OneDrive/Desktop/challenge 1/.venv/Scripts/Activate.ps1"


Ejecutar el script principal:

& "C:/Users/Alanl/OneDrive/Desktop/challenge 1/.venv/Scripts/python.exe" "challenge3/challange3.py"


Revisar la salida por consola y las gráficas generadas.
