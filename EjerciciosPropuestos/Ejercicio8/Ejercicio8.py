# ==============================================
# EJERCICIO 8: Clasificación con Random Forest – Fake News
# Dataset: noticias_procesadas.csv (generado en Ejercicio 3)
# Técnica: Random Forest para clasificación de texto
# ==============================================

# Fundamento teórico: Random Forest es un algoritmo de conjunto 
# (ensemble) que entrena múltiples árboles de decisión y combina sus 
# resultados para mejorar la precisión.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("=== EJERCICIO 8: RANDOM FOREST - FAKE NEWS ===\n")

# Paso 1: Cargar datos
print("Paso 1: Cargando datos de noticias_procesadas.csv...")
df = pd.read_csv('../Ejercicio3/noticias_procesadas.csv')
print(f"Dataset cargado: {df.shape[0]} noticias x {df.shape[1]} columnas")

# Paso 2: Vectorización con TF-IDF
print("\nPaso 2: Vectorizando texto con TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']
print(f"Matriz TF-IDF: {X.shape}")
print(f"Características extraídas: {X.shape[1]}")

# Paso 3: División de datos
print("\nPaso 3: Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# Paso 4: Entrenar modelo Random Forest
print("\nPaso 4: Entrenando Random Forest...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("Modelo Random Forest entrenado exitosamente")

# Paso 5: Evaluar modelo
print("\nPaso 5: Evaluando modelo...")
y_pred = model.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Importancia de características (top palabras)
print("\nTop 10 características más importantes:")
feature_names = vectorizer.get_feature_names_out()
feature_importance = pd.DataFrame({
    'Palabra': feature_names,
    'Importancia': model.feature_importances_
}).sort_values('Importancia', ascending=False).head(10)

for idx, row in feature_importance.iterrows():
    print(f"  '{row['Palabra']}': {row['Importancia']:.6f}")

print("\nEJERCICIO 8 COMPLETADO")