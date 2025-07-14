# ==============================================
# EJERCICIO 6: Visualización y análisis exploratorio extendido - Netflix
# Dataset: netflix_titles.csv
# Técnica: EDA (Exploratory Data Analysis) con visualizaciones avanzadas
# ==============================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=== EJERCICIO 6: ANÁLISIS EXPLORATORIO EXTENDIDO - NETFLIX ===\n")

# Paso 1: Carga de datos
print("Paso 1: Cargando dataset de Netflix...")
try:
    df = pd.read_csv('../../Dataset/netflix_titles.csv')
    print(f"Dataset cargado: {df.shape[0]:,} títulos x {df.shape[1]} columnas")
    print(f"Columnas disponibles: {list(df.columns)}")
except FileNotFoundError:
    print("Error: No se encontró el archivo netflix_titles.csv")
    print("Asegúrate de que el archivo esté en la carpeta Dataset/")
    exit()

print("\nVista previa del dataset:")
print(df.head())

print(f"\nInformación general del dataset:")
print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Tipos de datos:")
print(df.dtypes)

# Paso 2: Limpieza básica
print(f"\nPaso 2: Realizando limpieza básica...")

# Convertir date_added a datetime
print("Procesando fechas de adición...")
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Limpiar release_year
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

print(f"Fechas procesadas:")
print(f"  Rango de años de adición: {df['year_added'].min():.0f} - {df['year_added'].max():.0f}")
print(f"  Rango de años de lanzamiento: {df['release_year'].min():.0f} - {df['release_year'].max():.0f}")

# Análisis de valores faltantes
print(f"\nValores faltantes por columna:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
for col, missing in missing_data.items():
    if missing > 0:
        print(f"  {col}: {missing:,} ({missing_percent[col]:.1f}%)")

# Paso 3: Películas vs Series por año
print(f"\nPaso 3: Analizando distribución Movies vs TV Shows por año...")

# Filtrar datos válidos
df_valid = df.dropna(subset=['year_added', 'type'])

plt.figure(figsize=(14, 6))
sns.countplot(data=df_valid, x='year_added', hue='type', palette=['#E50914', '#B20710'])
plt.xticks(rotation=45)
plt.title('Títulos añadidos a Netflix por año según tipo', fontsize=16, fontweight='bold')
plt.xlabel('Año de adición a Netflix', fontsize=12)
plt.ylabel('Número de títulos', fontsize=12)
plt.legend(title='Tipo de contenido')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('netflix_movies_vs_shows_por_año.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'netflix_movies_vs_shows_por_año.png'")
plt.close()

# Estadísticas por año
print(f"\nEstadísticas de contenido por año:")
year_stats = df_valid.groupby(['year_added', 'type']).size().unstack(fill_value=0)
print(year_stats.tail())

# Paso 4: Distribución de duración de películas
print(f"\nPaso 4: Analizando duración de películas...")

# Filtrar solo películas
movie_df = df[df['type'] == 'Movie'].copy()
print(f"Total de películas en el dataset: {len(movie_df):,}")

# Extraer duración numérica
movie_df['duration_num'] = movie_df['duration'].str.replace(' min', '').astype('str')
movie_df['duration_num'] = pd.to_numeric(movie_df['duration_num'], errors='coerce')

# Filtrar valores válidos
movie_df_clean = movie_df.dropna(subset=['duration_num'])
print(f"Películas con duración válida: {len(movie_df_clean):,}")

# Estadísticas de duración
print(f"\nEstadísticas de duración de películas:")
print(f"  Duración promedio: {movie_df_clean['duration_num'].mean():.1f} minutos")
print(f"  Duración mediana: {movie_df_clean['duration_num'].median():.1f} minutos")
print(f"  Duración mínima: {movie_df_clean['duration_num'].min():.0f} minutos")
print(f"  Duración máxima: {movie_df_clean['duration_num'].max():.0f} minutos")
print(f"  Desviación estándar: {movie_df_clean['duration_num'].std():.1f} minutos")

# Crear histograma de duración
plt.figure(figsize=(10, 6))
sns.histplot(movie_df_clean['duration_num'], bins=30, color='#E50914', alpha=0.7)
plt.axvline(movie_df_clean['duration_num'].mean(), color='red', linestyle='--', 
            label=f'Promedio: {movie_df_clean["duration_num"].mean():.1f} min')
plt.axvline(movie_df_clean['duration_num'].median(), color='blue', linestyle='--', 
            label=f'Mediana: {movie_df_clean["duration_num"].median():.1f} min')
plt.title('Distribución de duración de películas en Netflix', fontsize=16, fontweight='bold')
plt.xlabel('Duración (minutos)', fontsize=12)
plt.ylabel('Número de películas', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('netflix_duracion_peliculas.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'netflix_duracion_peliculas.png'")
plt.close()

# Paso 5: Directores más frecuentes (Top 10)
print(f"\nPaso 5: Analizando directores más frecuentes...")

# Procesar directores (algunos títulos tienen múltiples directores)
directores_lista = []
for directores in df['director'].dropna():
    directores_separados = [director.strip() for director in str(directores).split(',')]
    directores_lista.extend(directores_separados)

print(f"Total de directores únicos encontrados: {len(set(directores_lista)):,}")

# Top 10 directores
top_directors = pd.Series(directores_lista).value_counts().head(10)
print(f"\nTop 10 directores más frecuentes:")
for i, (director, cantidad) in enumerate(top_directors.items(), 1):
    print(f"  {i:2d}. {director}: {cantidad} producciones")

# Crear gráfico horizontal
plt.figure(figsize=(10, 6))
top_directors.plot(kind='barh', color='#E50914', alpha=0.8)
plt.title('Top 10 directores con más producciones en Netflix', fontsize=16, fontweight='bold')
plt.xlabel('Cantidad de producciones', fontsize=12)
plt.ylabel('Director', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('netflix_top_directores.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'netflix_top_directores.png'")
plt.close()

# Análisis adicional: Países productores
print(f"\nAnálisis adicional: Países productores...")

# Procesar países
paises_lista = []
for paises in df['country'].dropna():
    paises_separados = [pais.strip() for pais in str(paises).split(',')]
    paises_lista.extend(paises_separados)

top_countries = pd.Series(paises_lista).value_counts().head(10)
print(f"\nTop 10 países productores:")
for i, (pais, cantidad) in enumerate(top_countries.items(), 1):
    print(f"  {i:2d}. {pais}: {cantidad} producciones")

# Gráfico de países
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar', color='#B20710', alpha=0.8)
plt.title('Top 10 países con más producciones en Netflix', fontsize=16, fontweight='bold')
plt.xlabel('País', fontsize=12)
plt.ylabel('Cantidad de producciones', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('netflix_top_paises.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'netflix_top_paises.png'")
plt.close()

# Análisis de géneros
print(f"\nAnálisis de géneros más populares...")

# Procesar géneros
generos_lista = []
for generos in df['listed_in'].dropna():
    generos_separados = [genero.strip() for genero in str(generos).split(',')]
    generos_lista.extend(generos_separados)

top_genres = pd.Series(generos_lista).value_counts().head(15)
print(f"\nTop 15 géneros más frecuentes:")
for i, (genero, cantidad) in enumerate(top_genres.items(), 1):
    print(f"  {i:2d}. {genero}: {cantidad} producciones")

# Gráfico de géneros
plt.figure(figsize=(14, 8))
top_genres.plot(kind='barh', color='#E50914', alpha=0.8)
plt.title('Top 15 géneros más populares en Netflix', fontsize=16, fontweight='bold')
plt.xlabel('Cantidad de producciones', fontsize=12)
plt.ylabel('Género', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('netflix_top_generos.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'netflix_top_generos.png'")
plt.close()

# Análisis temporal: Evolución del contenido
print(f"\nAnálisis temporal: Evolución del catálogo...")

# Contenido añadido por mes
monthly_content = df.groupby(['year_added', 'month_added']).size().reset_index(name='count')
monthly_content = monthly_content.dropna()

if not monthly_content.empty:
    plt.figure(figsize=(12, 6))
    pivot_monthly = monthly_content.pivot(index='month_added', columns='year_added', values='count')
    sns.heatmap(pivot_monthly, cmap='Reds', annot=False, fmt='d', cbar_kws={'label': 'Número de títulos'})
    plt.title('Mapa de calor: Títulos añadidos por mes y año', fontsize=16, fontweight='bold')
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('Mes', fontsize=12)
    plt.tight_layout()
    plt.savefig('netflix_heatmap_temporal.png', dpi=300, bbox_inches='tight')
    print("Mapa de calor guardado como 'netflix_heatmap_temporal.png'")
    plt.close()

# Resumen final de insights
print(f"\nRESUMEN DE INSIGHTS PRINCIPALES:")
print("=" * 50)
print(f"1. CONTENIDO TOTAL: {len(df):,} títulos en Netflix")
print(f"2. DISTRIBUCIÓN: {df['type'].value_counts()['Movie']:,} películas vs {df['type'].value_counts()['TV Show']:,} series")
print(f"3. DIRECTOR TOP: {top_directors.index[0]} con {top_directors.iloc[0]} producciones")
print(f"4. PAÍS TOP: {top_countries.index[0]} con {top_countries.iloc[0]} producciones")
print(f"5. GÉNERO TOP: {top_genres.index[0]} con {top_genres.iloc[0]} títulos")
print(f"6. DURACIÓN PROMEDIO: {movie_df_clean['duration_num'].mean():.0f} minutos para películas")
print(f"7. AÑO PICO: {year_stats.sum(axis=1).idxmax():.0f} (más títulos añadidos)")

print(f"\nARCHIVOS GENERADOS:")
print("- netflix_movies_vs_shows_por_año.png")
print("- netflix_duracion_peliculas.png") 
print("- netflix_top_directores.png")
print("- netflix_top_paises.png")
print("- netflix_top_generos.png")
print("- netflix_heatmap_temporal.png")

print(f"\nEJERCICIO 6: ANÁLISIS EXPLORATORIO COMPLETADO EXITOSAMENTE!")