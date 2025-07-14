# ==============================================
# EJERCICIO 5: Clasificación de texto con TF-IDF – Noticias falsas vs verdaderas
# Objetivo: Crear un modelo que clasifique si una noticia es verdadera o falsa en 
# base al texto (dataset noticias_procesadas.csv).
# ==============================================

# 1. Carga de datos
import pandas as pd
df = pd.read_csv('../Ejercicio3/noticias_procesadas.csv')

# 2. Vectorización con TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. División de datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Entrenamiento con Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluación del modelo
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))