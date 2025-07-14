# ==============================================
# Ejercicio 12: Detección de Spam – TF-IDF + SVM
# Objetivo: Clasificar correos como spam o no usando SVM.
# Dataset utilizado: spam.csv con columnas text, label
# ==============================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

print("=== EJERCICIO 12: DETECCIÓN DE SPAM - TF-IDF + SVM ===\n")

# Cargar datos con encoding correcto
print("Cargando dataset...")
df = pd.read_csv('../../Dataset/spam.csv', encoding='latin-1')

# Verificar estructura del dataset
print(f"Columnas encontradas: {list(df.columns)}")
print(f"Forma del dataset: {df.shape}")

# Limpiar dataset - el archivo original puede tener columnas extra vacías
# Mantener solo las dos primeras columnas que contienen los datos importantes
df = df.iloc[:, :2]  # Tomar solo las primeras 2 columnas
df.columns = ['label', 'text']  # Renombrar a nombres estándar

# Verificar datos
print(f"\nPrimeras líneas del dataset:")
print(df.head())

print(f"\nDistribución de etiquetas:")
print(df['label'].value_counts())

# Limpiar datos
df = df.dropna()  # Eliminar filas vacías

print(f"\nDatos después de limpieza: {df.shape[0]} registros")

# Vectorización con TF-IDF
print("\nVectorizando texto con TF-IDF...")
X = TfidfVectorizer(stop_words='english', 
                   max_features=3000).fit_transform(df['text'])
y = df['label']

print(f"Matriz TF-IDF: {X.shape}")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                   random_state=42)

print(f"Datos de entrenamiento: {X_train.shape[0]}")
print(f"Datos de prueba: {X_test.shape[0]}")

# Entrenar modelo SVM
print("\nEntrenando modelo SVM...")
model = SVC()
model.fit(X_train, y_train)

# Realizar predicciones
print("Realizando predicciones...")
y_pred = model.predict(X_test)

# Mostrar resultados
print("\n" + "="*50)
print("RESULTADOS DE CLASIFICACIÓN:")
print("="*50)
print(classification_report(y_test, y_pred))
