# ==============================================
# EJERCICIO 3: Carga y preparación de datos de texto con el dataset "Fake News"
# Dataset: Fake.csv y True.csv
# Objetivo: Cumplir exactamente los 6 puntos del enunciado
# ==============================================

# Paso 1: Importar bibliotecas necesarias
import pandas as pd # Para manejo de datos en forma de tablas
import numpy as np # Para operaciones numéricas

print("=== EJERCICIO 3: PROCESAMIENTO DE DATOS DE TEXTO - FAKE NEWS ===\n")

# PUNTO 1: Carga ambos archivos como dos DataFrames distintos
print("PUNTO 1: Cargando archivos Fake.csv y True.csv como DataFrames distintos:")
print("=" * 70)

# Cargar archivo de noticias falsas
try:
    df_fake = pd.read_csv("../../Dataset/fake.csv")
    print("Archivo 'fake.csv' cargado exitosamente")
    print(f"Noticias falsas: {df_fake.shape[0]:,} registros × {df_fake.shape[1]} columnas")
    print(f"Columnas en fake.csv: {list(df_fake.columns)}")
except FileNotFoundError:
    try:
        # Intentar con nombre en mayúscula
        df_fake = pd.read_csv("../../Dataset/Fake.csv")
        print("Archivo 'Fake.csv' cargado exitosamente")
        print(f"Noticias falsas: {df_fake.shape[0]:,} registros × {df_fake.shape[1]} columnas")
        print(f"Columnas en Fake.csv: {list(df_fake.columns)}")
    except FileNotFoundError:
        print("Error: No se encontró el archivo fake.csv o Fake.csv")
        print("Asegúrate de que el archivo esté en la carpeta Dataset/")
        exit()

# Cargar archivo de noticias verdaderas
try:
    df_true = pd.read_csv("../../Dataset/true.csv")
    print("Archivo 'true.csv' cargado exitosamente")
    print(f"Noticias verdaderas: {df_true.shape[0]:,} registros × {df_true.shape[1]} columnas")
    print(f"Columnas en true.csv: {list(df_true.columns)}")
except FileNotFoundError:
    try:
        # Intentar con nombre en mayúscula
        df_true = pd.read_csv("../../Dataset/True.csv")
        print("Archivo 'True.csv' cargado exitosamente")
        print(f"Noticias verdaderas: {df_true.shape[0]:,} registros × {df_true.shape[1]} columnas")
        print(f"Columnas en True.csv: {list(df_true.columns)}")
    except FileNotFoundError:
        print("Error: No se encontró el archivo true.csv o True.csv")
        print("Asegúrate de que el archivo esté en la carpeta Dataset/")
        exit()

# Mostrar muestra de ambos DataFrames
print(f"\nMuestra de noticias falsas (primeras 3 filas):")
print(df_fake.head(3))

print(f"\nMuestra de noticias verdaderas (primeras 3 filas):")
print(df_true.head(3))

# PUNTO 2: Agrega una columna label con valor 0 para noticias falsas y 1 para verdaderas
print(f"\nPUNTO 2: Agregando columna 'label':")
print("=" * 35)

# Crear copias para evitar modificar los originales
df_fake_labeled = df_fake.copy()
df_true_labeled = df_true.copy()

# Agregar columna label
df_fake_labeled['label'] = 0  # 0 = Noticia falsa
df_true_labeled['label'] = 1  # 1 = Noticia verdadera

print(f"Etiqueta agregada a noticias falsas: label = 0")
print(f"Etiqueta agregada a noticias verdaderas: label = 1")

# Verificar las etiquetas
print(f"\nVerificación de etiquetas:")
print(f"Noticias falsas con label=0: {(df_fake_labeled['label'] == 0).sum():,}")
print(f"Noticias verdaderas con label=1: {(df_true_labeled['label'] == 1).sum():,}")

print(f"\nColumnas después de agregar label:")
print(f"fake.csv: {list(df_fake_labeled.columns)}")
print(f"true.csv: {list(df_true_labeled.columns)}")

# PUNTO 3: Une ambos DataFrames en uno solo
print(f"\nPUNTO 3: Uniendo ambos DataFrames en uno solo:")
print("=" * 45)

# Concatenar los DataFrames
df_combined = pd.concat([df_fake_labeled, df_true_labeled], ignore_index=True)

print(f"DataFrames combinados exitosamente")
print(f"Total de registros: {df_combined.shape[0]:,}")
print(f"Total de columnas: {df_combined.shape[1]}")
print(f"Columnas en dataset combinado: {list(df_combined.columns)}")

# Verificar la distribución de etiquetas
distribucion_labels = df_combined['label'].value_counts().sort_index()
print(f"\nDistribución final de etiquetas:")
print(f"Noticias falsas (0): {distribucion_labels[0]:,} ({distribucion_labels[0]/len(df_combined)*100:.1f}%)")
print(f"Noticias verdaderas (1): {distribucion_labels[1]:,} ({distribucion_labels[1]/len(df_combined)*100:.1f}%)")

# PUNTO 4: Elimina las columnas subject y date, dejando solo text y label
print(f"\nPUNTO 4: Eliminando columnas subject y date, dejando solo text y label:")
print("=" * 75)

print(f"Columnas antes de eliminar: {list(df_combined.columns)}")

# Verificar qué columnas existen antes de eliminar
columnas_a_eliminar = []
if 'subject' in df_combined.columns:
    columnas_a_eliminar.append('subject')
    print(f"Columna 'subject' encontrada - será eliminada")
if 'date' in df_combined.columns:
    columnas_a_eliminar.append('date')
    print(f"Columna 'date' encontrada - será eliminada")

# Eliminar columnas especificadas
if columnas_a_eliminar:
    df_processed = df_combined.drop(columns=columnas_a_eliminar)
    print(f"Columnas eliminadas: {columnas_a_eliminar}")
else:
    df_processed = df_combined.copy()
    print(f"No se encontraron las columnas 'subject' o 'date'")

# Verificar si existen las columnas text y label
columnas_deseadas = ['text', 'label']
columnas_disponibles = [col for col in columnas_deseadas if col in df_processed.columns]

if len(columnas_disponibles) == 2:
    df_final = df_processed[columnas_deseadas].copy()
    print(f"Dataset final con columnas deseadas: {list(df_final.columns)}")
else:
    print(f"Advertencia: No se encontraron todas las columnas esperadas")
    print(f"Columnas disponibles: {list(df_processed.columns)}")
    
    # Adaptar a las columnas que existan
    if 'title' in df_processed.columns and 'text' not in df_processed.columns:
        df_final = df_processed[['title', 'label']].copy()
        df_final = df_final.rename(columns={'title': 'text'})
        print(f"Usando 'title' como 'text'")
    elif len(df_processed.columns) >= 2:
        # Tomar las últimas dos columnas (asumiendo que una es text-like y otra es label)
        text_col = [col for col in df_processed.columns if col != 'label'][0]
        df_final = df_processed[[text_col, 'label']].copy()
        if text_col != 'text':
            df_final = df_final.rename(columns={text_col: 'text'})
        print(f"Usando '{text_col}' como 'text'")
    else:
        df_final = df_processed.copy()

print(f"Columnas finales: {list(df_final.columns)}")
print(f"Tamaño final: {df_final.shape[0]:,} filas × {df_final.shape[1]} columnas")

# PUNTO 5: Verifica si hay valores nulos
print(f"\nPUNTO 5: Verificando valores nulos:")
print("=" * 35)

valores_nulos = df_final.isnull().sum()
print(f"Valores nulos por columna:")
total_nulos = 0
for columna, nulos in valores_nulos.items():
    if nulos > 0:
        porcentaje = (nulos / len(df_final)) * 100
        print(f"  {columna}: {nulos:,} valores nulos ({porcentaje:.2f}%)")
        total_nulos += nulos
    else:
        print(f"  {columna}: Sin valores nulos")

if total_nulos > 0:
    print(f"\nTotal de valores nulos encontrados: {total_nulos:,}")
    print("Eliminando filas con valores nulos...")
    filas_antes = len(df_final)
    df_final = df_final.dropna()
    filas_despues = len(df_final)
    print(f"Filas eliminadas: {filas_antes - filas_despues:,}")
    print(f"Filas restantes: {filas_despues:,}")
else:
    print(f"\nNo se encontraron valores nulos - dataset está completo")

# PUNTO 6: Guarda el nuevo dataset como noticias_procesadas.csv
print(f"\nPUNTO 6: Guardando dataset como noticias_procesadas.csv:")
print("=" * 55)

archivo_salida = "noticias_procesadas.csv"
try:
    df_final.to_csv(archivo_salida, index=False)
    print(f"Dataset guardado exitosamente como '{archivo_salida}'")
    print(f"Ubicación: directorio actual")
    print(f"Tamaño del archivo: {len(df_final):,} filas × {len(df_final.columns)} columnas")
    
    # Información del archivo guardado
    print(f"\nInformación del archivo guardado:")
    print(f"Nombre: {archivo_salida}")
    print(f"Columnas: {list(df_final.columns)}")
    print(f"Registros totales: {len(df_final):,}")
    
    if 'label' in df_final.columns:
        dist_final = df_final['label'].value_counts().sort_index()
        print(f"Distribución final:")
        print(f"  Noticias falsas (0): {dist_final[0]:,}")
        print(f"  Noticias verdaderas (1): {dist_final[1]:,}")
    
except Exception as e:
    print(f"Error al guardar el archivo: {e}")

# Mostrar muestra del dataset final
print(f"\nMuestra del dataset procesado final:")
print("=" * 40)
for i, row in df_final.head(3).iterrows():
    texto_col = 'text' if 'text' in df_final.columns else df_final.columns[0]
    label_val = "VERDADERA" if row['label'] == 1 else "FALSA"
    print(f"Noticia {i+1} ({label_val}):")
    print(f"  {row[texto_col][:100]}...")
    print()



# Resumen del procesamiento realizado
print(f"\nRESUMEN DEL PROCESAMIENTO REALIZADO:")
print("=" * 40)
print(f"• Archivos originales procesados: fake.csv y true.csv")
print(f"• Dataset final: {len(df_final):,} noticias etiquetadas")
print(f"• Formato final: texto + etiqueta binaria")
print(f"• Archivo generado: {archivo_salida}")
print(f"• Dataset listo para análisis de texto o machine learning")

print(f"\nEJERCICIO 3 COMPLETADO EXITOSAMENTE!")