# ==============================================
# EJERCICIO 9: Detección de Anomalías – Netflix
# Dataset: netflix_titles.csv
# Técnica: Isolation Forest para detección de anomalías
# ==============================================

# Fundamento teórico: Isolation Forest es un algoritmo no supervisado que 
# identifica observaciones anómalas en un conjunto de datos, ideal para 
# grandes volúmenes.

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

print("=== EJERCICIO 9: DETECCIÓN DE ANOMALÍAS - NETFLIX ===\n")

# Paso 1: Cargar datos
print("Paso 1: Cargando datos de netflix_titles.csv...")
df = pd.read_csv('../../Dataset/netflix_titles.csv')
print(f"Dataset cargado: {df.shape[0]} títulos x {df.shape[1]} columnas")

# Paso 2: Filtrar películas
print("\nPaso 2: Filtrando películas...")
movie_df = df[df['type'] == 'Movie']
print(f"Películas encontradas: {movie_df.shape[0]}")

# Paso 3: Limpiar datos de duración
print("\nPaso 3: Limpiando datos de duración...")
movie_df = movie_df.dropna(subset=['duration'])
movie_df = movie_df.copy()
movie_df['duration_num'] = movie_df['duration'].str.replace(' min', '').astype(float)
print(f"Películas con duración válida: {movie_df.shape[0]}")
print(f"Duración promedio: {movie_df['duration_num'].mean():.1f} minutos")
print(f"Duración min-max: {movie_df['duration_num'].min():.0f} - {movie_df['duration_num'].max():.0f} minutos")

# Paso 4: Detectar anomalías
print("\nPaso 4: Detectando anomalías con Isolation Forest...")
model = IsolationForest(contamination=0.02)
movie_df['anomaly'] = model.fit_predict(movie_df[['duration_num']])
print("Modelo Isolation Forest entrenado")

# Contar anomalías
anomalias = (movie_df['anomaly'] == -1).sum()
normales = (movie_df['anomaly'] == 1).sum()
print(f"Películas normales: {normales}")
print(f"Películas anómalas: {anomalias}")

# Paso 5: Visualizar resultados
print("\nPaso 5: Visualizando resultados...")
plt.figure(figsize=(12, 6))
sns.histplot(data=movie_df, x='duration_num', hue='anomaly', palette='Set2', bins=30)
plt.title('Películas normales vs anómalas según duración')
plt.xlabel('Duración (minutos)')
plt.ylabel('Cantidad de películas')
plt.legend(labels=['Anómala', 'Normal'])
plt.savefig('anomalias_netflix.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'anomalias_netflix.png'")
plt.close()

# Paso 6: Mostrar películas anómalas
print("\nPaso 6: Películas con duración anómala:")
print("="*60)
anomalas = movie_df[movie_df['anomaly'] == -1][['title', 'duration', 'duration_num']].sort_values('duration_num')
for idx, row in anomalas.iterrows():
    print(f"  {row['title'][:50]:<50} - {row['duration']}")

print(f"\nRango de duraciones anómalas: {anomalas['duration_num'].min():.0f} - {anomalas['duration_num'].max():.0f} minutos")

print("\nEJERCICIO 9 COMPLETADO")