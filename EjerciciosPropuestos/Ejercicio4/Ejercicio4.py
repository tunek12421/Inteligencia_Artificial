# ==============================================
# EJERCICIO 4: Modelo de clasificación binaria - Titanic Survivors
# Dataset: titanic_limpio_ml.csv (generado en Ejercicio 1-4)
# Técnica: Regresión Logística para clasificación binaria
# ==============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("=== EJERCICIO 4: MODELO DE CLASIFICACIÓN BINARIA - TITANIC SURVIVORS ===\n")

# Paso 1: Carga de datos
print("Paso 1: Cargando datos del archivo limpio para ML...")
try:
    df = pd.read_csv('../Ejercicio1_4/titanic_limpio_ml.csv')
    print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Columnas disponibles: {list(df.columns)}")
except FileNotFoundError:
    print("Error: No se encontró el archivo titanic_limpio_ml.csv")
    print("Ejecuta primero el Ejercicio 1-4 para generar el archivo")
    exit()

print("\nVista previa del dataset:")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)

# Identificar la columna de supervivencia
survived_cols = [col for col in df.columns if 'urvived' in col]
if survived_cols:
    target_col = survived_cols[0]
    print(f"\nColumna objetivo identificada: '{target_col}'")
else:
    print("Error: No se encontró columna de supervivencia")
    exit()

# Paso 2: Separación de variables
print(f"\nPaso 2: Separando variables...")
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"Variables predictoras (X): {X.shape[1]} columnas")
print(f"Nombres de características: {list(X.columns)}")
print(f"Variable objetivo (y): '{target_col}' con distribución:")
distribucion = y.value_counts().sort_index()
for valor, cantidad in distribucion.items():
    estado = "Sobrevivió" if valor == 1 else "No sobrevivió"
    porcentaje = (cantidad / len(y)) * 100
    print(f"  {estado} ({valor}): {cantidad:,} pasajeros ({porcentaje:.1f}%)")

# Paso 3: División en entrenamiento y prueba
print(f"\nPaso 3: Dividiendo datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Datos de entrenamiento: {X_train.shape[0]:,} muestras")
print(f"Datos de prueba: {X_test.shape[0]:,} muestras")
print(f"Proporción de datos de prueba: {0.3*100:.0f}%")

# Paso 4: Entrenar modelo de Regresión Logística
print(f"\nPaso 4: Entrenando modelo de Regresión Logística...")
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
print("Modelo entrenado exitosamente")

# Paso 5: Realizar predicciones
print(f"\nPaso 5: Realizando predicciones...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Paso 6: Evaluar el modelo
print(f"\nPaso 6: Evaluando el modelo...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\nReporte de clasificación detallado:")
print(classification_report(y_test, y_pred, target_names=['No sobrevivió', 'Sobrevivió']))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No sobrevivió', 'Sobrevivió'], 
            yticklabels=['No sobrevivió', 'Sobrevivió'])
plt.title('Matriz de confusión - Titanic Survivors')
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.tight_layout()
plt.savefig('confusion_matrix_titanic_survivors.png', dpi=300, bbox_inches='tight')
print("Matriz de confusión guardada como 'confusion_matrix_titanic_survivors.png'")
plt.close()

# Análisis de importancia de características
print(f"\nAnálisis de importancia de características:")
feature_importance = pd.DataFrame({
    'Característica': X.columns,
    'Coeficiente': model.coef_[0],
    'Importancia_Abs': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importancia_Abs', ascending=False)

print("\nRanking de características más importantes:")
print("-" * 50)
for idx, row in feature_importance.iterrows():
    direction = "Aumenta" if row['Coeficiente'] > 0 else "Disminuye"
    print(f"{row['Característica']:15}: {row['Coeficiente']:7.3f} ({direction} supervivencia)")

# Predicciones de ejemplo
print(f"\nEjemplos de predicciones:")
print("Real        | Predicho    | Probabilidad")
print("-" * 45)
for i in range(min(10, len(y_test))):
    real = "Sobrevivió" if y_test.iloc[i] == 1 else "No sobrevivió"
    pred = "Sobrevivió" if y_pred[i] == 1 else "No sobrevivió"
    prob = y_pred_proba[i][1]
    print(f"{real:11} | {pred:11} | {prob:.3f}")

# Análisis de la matriz de confusión
tn, fp, fn, tp = conf_matrix.ravel()
print(f"\nAnálisis detallado de la matriz de confusión:")
print(f"Verdaderos Negativos (TN): {tn} - Predijo 'No sobrevivió' y era correcto")
print(f"Falsos Positivos (FP): {fp} - Predijo 'Sobrevivió' pero era incorrecto")
print(f"Falsos Negativos (FN): {fn} - Predijo 'No sobrevivió' pero era incorrecto")
print(f"Verdaderos Positivos (TP): {tp} - Predijo 'Sobrevivió' y era correcto")

# Métricas adicionales
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMétricas adicionales:")
print(f"Precisión (Precision): {precision:.4f} - De los que predijo como supervivientes, cuántos eran correctos")
print(f"Recall (Sensibilidad): {recall:.4f} - De los supervivientes reales, cuántos detectó")
print(f"F1-Score: {f1_score:.4f} - Media armónica de precisión y recall")

# Estadísticas finales
print(f"\nESTADÍSTICAS FINALES:")
print("=" * 30)
print(f"Total de pasajeros analizados: {len(y_test):,}")
print(f"Predicciones correctas: {(y_test == y_pred).sum():,}")
print(f"Predicciones incorrectas: {(y_test != y_pred).sum():,}")
print(f"Precisión final: {accuracy:.4f}")

# Variables más influyentes
print(f"\nVARIABLES MÁS INFLUYENTES:")
print("-" * 30)
top_3 = feature_importance.head(3)
for idx, row in top_3.iterrows():
    direction = "favorece" if row['Coeficiente'] > 0 else "reduce"
    print(f"{idx+1}. {row['Característica']}: {direction} la supervivencia")

print(f"\nModelo de clasificación binaria completado exitosamente")
print(f"Archivo generado: confusion_matrix_titanic_survivors.png")