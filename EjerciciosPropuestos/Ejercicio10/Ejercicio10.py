# ==============================================
# EJERCICIO 10: Comparación entre modelos de clasificación – Titanic
# Objetivo: Comparar Regresión Logística, Árbol de Decisión y Random Forest
# ==============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("=== EJERCICIO 10: COMPARACIÓN DE MODELOS - TITANIC ===\n")

# Cargar datos
df = pd.read_csv('../Ejercicio1/titanic_limpio.csv')
X = df.drop('2urvived', axis=1)
y = df['2urvived']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("MODELO 1: REGRESIÓN LOGÍSTICA")
print("-" * 35)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)
print(f"Precisión: {log_accuracy:.4f}")
print(classification_report(y_test, log_pred))

print("\nMODELO 2: ÁRBOL DE DECISIÓN")
print("-" * 32)
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
print(f"Precisión: {tree_accuracy:.4f}")
print(classification_report(y_test, tree_pred))

print("\nMODELO 3: RANDOM FOREST")
print("-" * 25)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Precisión: {rf_accuracy:.4f}")
print(classification_report(y_test, rf_pred))

print("\nCOMPARACIÓN FINAL:")
print("=" * 20)
results = [
    ("Regresión Logística", log_accuracy),
    ("Árbol de Decisión", tree_accuracy),
    ("Random Forest", rf_accuracy)
]

results.sort(key=lambda x: x[1], reverse=True)
for i, (model, accuracy) in enumerate(results, 1):
    print(f"{i}. {model}: {accuracy:.4f}")

print(f"\nEl mejor modelo fue: {results[0][0]}")