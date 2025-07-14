# ===============================
# Ejercicio 4: Modelo de clasificación binaria – Titanic Survivors
# ===============================
# Objetivo: Predecir si un pasajero sobrevivió o no en base a características del dataset Titanic ya limpiado (titanic_limpio.csv).

# 1. Carga de datos
import pandas as pd
df = pd.read_csv('../Ejercicio1/titanic_limpio.csv')

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


# ===============================
# Ejercicio 5: Clasificación de texto con TF-IDF – Noticias falsas vs verdaderas
# ===============================
# Objetivo: Crear un modelo que clasifique si una noticia es verdadera o falsa en base al texto (dataset noticias_procesadas.csv).

# 1. Carga de datos
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


# ===============================
# Ejercicio 6: Visualización y análisis exploratorio extendido – Netflix
# ===============================
# Objetivo: Profundizar en análisis exploratorio de datos visuales y patrones en las producciones de Netflix (usando netflix_titles.csv).

# 1. Carga de datos
df = pd.read_csv('../../Dataset/netflix_titles.csv')

# 2. Limpieza básica
df['date_added'] = pd.to_datetime(df['date_added'])
df['year_added'] = df['date_added'].dt.year

# 3. Películas vs Series por año
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=df, x='year_added', hue='type')
plt.xticks(rotation=45)
plt.title('Títulos añadidos por año según tipo')
plt.show()

# 4. Distribución de duración de películas
movie_df = df[df['type'] == 'Movie']
movie_df['duration_num'] = movie_df['duration'].str.replace(' min', '').astype(float)
sns.histplot(movie_df['duration_num'], bins=30)
plt.title('Duración de películas en minutos')
plt.show()

# 5. Directores más frecuentes (Top 10)
top_directors = df['director'].value_counts().head(10)
top_directors.plot(kind='barh', title='Top 10 directores en Netflix')
plt.xlabel('Cantidad de producciones')
plt.show()

# ===============================
# Preguntas para responder después de resolver los ejercicios:
# ===============================
# • ¿Qué variables influyeron más en el modelo?
# • ¿Qué métricas fueron mejores y por qué?
# • ¿Cómo mejorarías este modelo para un caso real?