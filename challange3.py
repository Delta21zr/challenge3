# -----------------------------
# Proyecto completo de predicción de churn
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# -----------------------------
# 1 Cargar CSV limpio
# -----------------------------
ruta_csv = r"C:\Users\Alanl\OneDrive\Desktop\challenge 1\challange2\TelecomX_Datos_Limpios.csv"
df = pd.read_csv(ruta_csv)

print("Primeras filas del dataset:")
print(df.head(), "\n")

print("Columnas del dataset:")
print(df.columns.tolist())

# -----------------------------
# 2️Crear variable objetivo binaria
# Usamos 'Churn_Yes' del one-hot encoding si no existe 'Churn_Binario'
# -----------------------------
if 'Churn_Binario' not in df.columns:
    if 'Churn_Yes' in df.columns:
        df['Churn_Binario'] = df['Churn_Yes']
    else:
        raise Exception("No se encontró ninguna columna de Churn en el CSV.")

# -----------------------------
# 3 Separar características (X) y variable objetivo (y)
# -----------------------------
# Eliminamos columnas que no aportan valor predictivo
columnas_a_eliminar = ['customerID'] if 'customerID' in df.columns else []
X = df.drop(columns=columnas_a_eliminar + ['Churn_Binario'])
y = df['Churn_Binario']

print(f"\nNúmero de filas y columnas en X: {X.shape}")
print(f"\nNúmero de observaciones en y: {y.shape}")

# -----------------------------
# 4 Evaluar desbalance de clases
# -----------------------------
prop_churn = y.value_counts(normalize=True)
print("\nProporción de clases (Churn):")
print(prop_churn)

# -----------------------------
# 5 Normalización de datos numéricos (para modelos sensibles a escala)
# -----------------------------
numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numericas] = scaler.fit_transform(X[numericas])

# -----------------------------
# 6 División entrenamiento/prueba
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ----------------------------
# 7 Entrenamiento de modelos
# -----------------------------

# Modelo 1: Regresión Logística
modelo_lr = LogisticRegression(max_iter=1000)
modelo_lr.fit(X_train, y_train)
y_pred_lr = modelo_lr.predict(X_test)

# Modelo 2: K-Nearest Neighbors
modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(X_train, y_train)
y_pred_knn = modelo_knn.predict(X_test)

# Modelo 3: Random Forest (no necesita normalización)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train_rf, y_train_rf)
y_pred_rf = modelo_rf.predict(X_test_rf)

# -----------------------------
# 8Evaluación de modelos
# -----------------------------
def evaluar_modelo(y_true, y_pred, nombre_modelo):
    print(f"\n--- Evaluación: {nombre_modelo} ---")
    print("Exactitud:", accuracy_score(y_true, y_pred))
    print("Precisión:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_true, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred))

evaluar_modelo(y_test, y_pred_lr, "Regresión Logística")
evaluar_modelo(y_test, y_pred_knn, "K-Nearest Neighbors")
evaluar_modelo(y_test_rf, y_pred_rf, "Random Forest")

# -----------------------------
# 9 Análisis de importancia de variables
# -----------------------------
# Para regresión logística
coef_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Coeficiente': modelo_lr.coef_[0]
}).sort_values(by='Coeficiente', key=abs, ascending=False)
print("\nVariables más relevantes según Regresión Logística:")
print(coef_df.head(10))

# Para Random Forest
importancias_rf = pd.DataFrame({
    'Variable': X_train_rf.columns,
    'Importancia': modelo_rf.feature_importances_
}).sort_values(by='Importancia', ascending=False)
print("\nVariables más relevantes según Random Forest:")
print(importancias_rf.head(10))

# -----------------------------
# Matriz de correlación
# -----------------------------
plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title("Matriz de correlación entre variables numéricas")
plt.show()

# -----------------------------
#  Boxplots para explorar relación de variables numéricas con Churn
# -----------------------------
numericas_importantes = ['customer.tenure', 'account.Charges.Total', 'account.Charges.Monthly']
for col in numericas_importantes:
    if col in df.columns:
        plt.figure(figsize=(8,5))
        sns.boxplot(x='Churn_Binario', y=col, data=df)
        plt.title(f'{col} vs Churn')
        plt.show()
