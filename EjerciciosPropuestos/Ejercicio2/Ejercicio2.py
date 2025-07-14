# ==============================================
# EJERCICIO 2: Visualización de datos con el dataset "Netflix Movies and TV Shows"
# Dataset: Netflix dataset de Kaggle
# Objetivo: Cumplir exactamente los 5 puntos del enunciado
# ==============================================

# Paso 1: Importar bibliotecas necesarias
import pandas as pd # Para manejo de datos en forma de tablas
import numpy as np # Para operaciones numéricas
import matplotlib.pyplot as plt # Gráficos
import seaborn as sns # Visualizaciones
from collections import Counter # Para análisis de frecuencias

print("=== EJERCICIO 2: VISUALIZACIÓN DE DATOS - NETFLIX ===\n")

# PUNTO 1: Carga el archivo netflix_titles.csv
try:
    df = pd.read_csv("../../Dataset/netflix_titles.csv")
    print("PUNTO 1: Archivo netflix_titles.csv cargado exitosamente")
    print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"Columnas disponibles: {list(df.columns)}")
except FileNotFoundError:
    print("Error: No se encontró el archivo netflix_titles.csv")
    print("Asegúrate de que el archivo esté en la carpeta Dataset/")
    exit()

print("\nVista previa de los datos:")
print(df.head())

# PUNTO 2: Muestra cuántos títulos son películas y cuántos son programas de TV
print(f"\nPUNTO 2: Distribución de títulos por tipo:")
print("=" * 45)
tipo_contenido = df['type'].value_counts()
print("Conteo por tipo:")
print(tipo_contenido)

print(f"\nDistribución detallada:")
total_titulos = len(df)
for tipo, cantidad in tipo_contenido.items():
    porcentaje = (cantidad / total_titulos) * 100
    print(f"  {tipo}: {cantidad:,} títulos ({porcentaje:.1f}%)")

# PUNTO 3: Crea un gráfico de barras que muestre cuántas producciones se lanzaron por año
print(f"\nPUNTO 3: Creando gráfico de barras por año de lanzamiento...")
print("=" * 60)

# Limpiar y preparar datos de año
df_clean = df.copy()
if 'release_year' in df_clean.columns:
    df_clean['release_year'] = pd.to_numeric(df_clean['release_year'], errors='coerce')
    df_clean = df_clean.dropna(subset=['release_year'])
    
    # Contar producciones por año
    producciones_por_año = df_clean['release_year'].value_counts().sort_index()
    
    # Crear el gráfico de barras
    plt.figure(figsize=(15, 6))
    plt.bar(producciones_por_año.index, producciones_por_año.values, color='#E50914', alpha=0.7)
    plt.title('Producciones de Netflix por Año de Lanzamiento', fontsize=16, fontweight='bold')
    plt.xlabel('Año de Lanzamiento', fontsize=12)
    plt.ylabel('Número de Producciones', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('netflix_producciones_por_año.png', dpi=300, bbox_inches='tight')
    print("Gráfico de barras guardado como 'netflix_producciones_por_año.png'")
    plt.close()
    
    # Mostrar estadísticas del año
    año_mas_producciones = producciones_por_año.idxmax()
    max_producciones = producciones_por_año.max()
    print(f"Año con más lanzamientos: {int(año_mas_producciones)} ({max_producciones} producciones)")
    
else:
    print("No se encontró columna 'release_year' en el dataset")

# PUNTO 4: Visualiza con un countplot de Seaborn cuántas producciones hay por país (top 10)
print(f"\nPUNTO 4: Creando countplot de países (top 10)...")
print("=" * 50)

if 'country' in df.columns:
    # Preparar datos de países (algunos títulos tienen múltiples países separados por comas)
    paises_lista = []
    for paises in df['country'].dropna():
        # Dividir por comas y limpiar espacios
        paises_separados = [pais.strip() for pais in str(paises).split(',')]
        paises_lista.extend(paises_separados)
    
    # Contar y obtener top 10
    paises_contador = Counter(paises_lista)
    top10_paises = dict(paises_contador.most_common(10))
    
    print(f"TOP 10 PAÍSES CON MÁS PRODUCCIONES:")
    for i, (pais, cantidad) in enumerate(top10_paises.items(), 1):
        print(f"{i:2d}. {pais}: {cantidad} producciones")
    
    # Crear DataFrame para countplot
    df_paises = pd.DataFrame({'country': paises_lista})
    
    # Crear countplot con Seaborn
    plt.figure(figsize=(12, 6))
    paises_names = list(top10_paises.keys())
    
    sns.countplot(data=df_paises[df_paises['country'].isin(paises_names)], 
                  x='country', 
                  order=paises_names, 
                  palette='viridis')
    plt.title('Top 10 Países con Más Producciones en Netflix', fontsize=16, fontweight='bold')
    plt.xlabel('País', fontsize=12)
    plt.ylabel('Número de Producciones', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('netflix_top10_paises_countplot.png', dpi=300, bbox_inches='tight')
    print("Countplot de países guardado como 'netflix_top10_paises_countplot.png'")
    plt.close()
    
else:
    print("No se encontró columna 'country' en el dataset")

# PUNTO 5: Genera un gráfico de torta de los 5 directores más frecuentes
print(f"\nPUNTO 5: Creando gráfico de torta de directores (top 5)...")
print("=" * 58)

if 'director' in df.columns:
    # Preparar datos de directores
    directores_lista = []
    for directores in df['director'].dropna():
        # Algunos títulos tienen múltiples directores
        directores_separados = [director.strip() for director in str(directores).split(',')]
        directores_lista.extend(directores_separados)
    
    # Contar y obtener top 5
    directores_contador = Counter(directores_lista)
    top5_directores = dict(directores_contador.most_common(5))
    
    print(f"TOP 5 DIRECTORES MÁS FRECUENTES:")
    for i, (director, cantidad) in enumerate(top5_directores.items(), 1):
        print(f"{i}. {director}: {cantidad} producciones")
    
    # Crear gráfico de torta
    plt.figure(figsize=(10, 8))
    colores = ['#E50914', '#F40612', '#FF6B6B', '#FF8E8E', '#FFB1B1']
    wedges, texts, autotexts = plt.pie(top5_directores.values(), 
                                       labels=top5_directores.keys(), 
                                       autopct='%1.1f%%',
                                       colors=colores,
                                       startangle=90,
                                       explode=(0.05, 0, 0, 0, 0))  # Resaltar el primero
    
    plt.title('Top 5 Directores Más Frecuentes en Netflix', fontsize=16, fontweight='bold', pad=20)
    
    # Mejorar la apariencia del texto
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.axis('equal')  # Para que la torta sea circular
    plt.tight_layout()
    plt.savefig('netflix_top5_directores_torta.png', dpi=300, bbox_inches='tight')
    print("Gráfico de torta guardado como 'netflix_top5_directores_torta.png'")
    plt.close()
    
else:
    print("No se encontró columna 'director' en el dataset")

# Estadísticas adicionales del dataset
print(f"\nESTADÍSTICAS ADICIONALES DEL DATASET:")
print("=" * 40)

# Información general
print(f"Total de títulos únicos: {df['title'].nunique():,}")
print(f"Total de directores únicos: {df['director'].nunique():,}")

if 'country' in df.columns:
    paises_unicos = len(set(paises_lista)) if 'paises_lista' in locals() else df['country'].nunique()
    print(f"Total de países representados: {paises_unicos}")

if 'release_year' in df.columns:
    year_min = df['release_year'].min()
    year_max = df['release_year'].max()
    print(f"Rango de años: {int(year_min)} - {int(year_max)}")


# Resumen de archivos generados
print(f"\nARCHIVOS GENERADOS:")
print("- netflix_producciones_por_año.png (gráfico de barras)")
print("- netflix_top10_paises_countplot.png (countplot de países)")
print("- netflix_top5_directores_torta.png (gráfico de torta)")

print(f"\nEJERCICIO 2 COMPLETADO EXITOSAMENTE!")