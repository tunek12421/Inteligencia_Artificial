# Ejercicio 4: Modelo de clasificación binaria – Titanic Survivors
# Objetivo: Predecir si un pasajero sobrevivió o no en base a características del 
# dataset Titanic ya limpiado (titanic_limpio.csv).

# 1. Carga de datos
import pandas as pd
df = pd.read_csv('../Ejercicio1_4/titanic_limpio_ml.csv')

# 2. Separación de variables
X = df.drop('Survived', axis=1)
y = df['Survived']

# 3. División en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Entrenar un modelo de clasificación (Regresión Logística)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Evaluar el modelo
from sklearn.metrics import classification_report, accuracy_score
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print('Precisión:', accuracy_score(y_test, y_pred))