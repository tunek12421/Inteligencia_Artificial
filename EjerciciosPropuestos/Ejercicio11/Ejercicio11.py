# ==============================================
# EJERCICIO 11: Análisis de sentimiento con TF-IDF – Opiniones de productos
# Objetivo: Clasificar opiniones como positivas o negativas
# ==============================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("=== EJERCICIO 11: ANÁLISIS DE SENTIMIENTO - ROTTEN TOMATOES ===\n")

# Cargar datos
df = pd.read_csv('../../Dataset/data_rt.csv')
print(f"Dataset cargado: {df.shape[0]} reseñas")
print(f"Distribución de etiquetas:")
print(df['labels'].value_counts())

# Vectorización con TF-IDF
print("\nVectorizando texto con TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, min_df=2)
X = vectorizer.fit_transform(df['reviews'])
y = df['labels']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nMODELO 1: NAIVE BAYES")
print("-" * 23)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
print(f"Precisión: {nb_accuracy:.4f}")
print(classification_report(y_test, nb_pred, target_names=['Negativo', 'Positivo']))

print("\nMODELO 2: REGRESIÓN LOGÍSTICA")
print("-" * 35)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)
print(f"Precisión: {log_accuracy:.4f}")
print(classification_report(y_test, log_pred, target_names=['Negativo', 'Positivo']))

print("\nMODELO 3: RANDOM FOREST")
print("-" * 25)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Precisión: {rf_accuracy:.4f}")
print(classification_report(y_test, rf_pred, target_names=['Negativo', 'Positivo']))

print("\nCOMPARACIÓN FINAL:")
print("=" * 20)
results = [
    ("Naive Bayes", nb_accuracy),
    ("Regresión Logística", log_accuracy),
    ("Random Forest", rf_accuracy)
]

results.sort(key=lambda x: x[1], reverse=True)
for i, (model, accuracy) in enumerate(results, 1):
    print(f"{i}. {model}: {accuracy:.4f}")

print(f"\nEl mejor modelo fue: {results[0][0]}")