# ==============================================
# EJERCICIO 6: Visualización y análisis exploratorio extendido – Netflix
# Objetivo: Profundizar en análisis exploratorio de datos visuales y patrones en las 
# producciones de Netflix (usando netflix_titles.csv).
# ==============================================

# 1. Carga de datos
import pandas as pd
df = pd.read_csv('../../Dataset/netflix_titles.csv')

# 2. Limpieza básica
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
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

# Nota: Una vez resuelto los ejercicios se debe responder las siguientes preguntas:
# • ¿Qué variables influyeron más en el modelo?
# • ¿Qué métricas fueron mejores y por qué?
# • ¿Cómo mejorarías este modelo para un caso real?