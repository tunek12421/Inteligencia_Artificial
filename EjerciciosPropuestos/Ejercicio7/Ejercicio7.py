# ==============================================
# EJERCICIO 7: Árboles de Decisión – Titanic
# Dataset: titanic_limpio.csv (generado en Ejercicio 1)
# Técnica: Decision Tree para clasificación binaria
# ==============================================

# Fundamento teórico: Los Árboles de Decisión son algoritmos supervisados 
# que utilizan estructuras ramificadas para representar decisiones basadas en 
# reglas. Son interpretables y adecuados para problemas de clasificación.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print("=== EJERCICIO 7: ÁRBOLES DE DECISIÓN - TITANIC ===\n")

# Paso 1: Cargar datos
print("Paso 1: Cargando datos de titanic_limpio.csv...")
df = pd.read_csv('../Ejercicio1/titanic_limpio.csv')
print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

# Paso 2: Separar variables
print("\nPaso 2: Separando variables...")
X = df.drop('2urvived', axis=1)
y = df['2urvived']
print(f"Variables predictoras: {X.shape[1]}")
print(f"Variable objetivo: Survived")

# Paso 3: División de datos
print("\nPaso 3: Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# Paso 4: Entrenar modelo
print("\nPaso 4: Entrenando árbol de decisión...")
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
print("Modelo entrenado exitosamente")

# Paso 5: Visualizar árbol
print("\nPaso 5: Visualizando árbol de decisión...")
plt.figure(figsize=(15,10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Sí'], filled=True)
plt.title('Árbol de Decisión - Supervivencia Titanic')
plt.savefig('arbol_decision_titanic.png', dpi=300, bbox_inches='tight')
print("Árbol guardado como 'arbol_decision_titanic.png'")
plt.close()

# Paso 6: Evaluar modelo
print("\nPaso 6: Evaluando modelo...")
y_pred = model.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Importancia de características
print("\nImportancia de características:")
feature_importance = pd.DataFrame({
    'Característica': X.columns,
    'Importancia': model.feature_importances_
}).sort_values('Importancia', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"  {row['Característica']}: {row['Importancia']:.4f}")

print("\nEJERCICIO 7 COMPLETADO")