# ==============================================
# EJERCICIO 1: Exploración y limpieza de datos con el dataset "Titanic"
# Dataset: Titanic - train.csv from Kaggle
# Objetivo: Cumplir exactamente los 6 puntos del enunciado
# ==============================================

# Paso 1: Importar bibliotecas necesarias
import pandas as pd # Para manejo de datos en forma de tablas
import numpy as np # Para operaciones numéricas

print("=== EJERCICIO 1: EXPLORACIÓN Y LIMPIEZA DE DATOS - TITANIC ===\n")

# PUNTO 1: Carga el archivo CSV con pandas
try:
    df = pd.read_csv("../../Dataset/train.csv", sep=';')
    print("PUNTO 1: Archivo train.csv cargado exitosamente con pandas")
    
    # Si solo hay una columna, cambiar separador
    if df.shape[1] == 1:
        df = pd.read_csv("../../Dataset/train.csv", sep=',')
        
except FileNotFoundError:
    print("Error: No se encontró el archivo train.csv")
    print("Asegúrate de que el archivo esté en la carpeta Dataset/")
    exit()

# Limpiar el dataset - eliminar columnas 'zero' si existen
important_cols = [col for col in df.columns if 'zero' not in col.lower()]
df = df[important_cols].copy()

print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"Columnas disponibles: {list(df.columns)}")

# PUNTO 2: Muestra las primeras 10 filas con .head()
print(f"\nPUNTO 2: Primeras 10 filas del dataset:")
print("=" * 60)
print(df.head(10))  # ESPECÍFICAMENTE 10 filas como pide el enunciado

# PUNTO 3: Calcula cuántos valores faltantes hay por columna
print(f"\nPUNTO 3: Valores faltantes por columna:")
print("=" * 45)
valores_faltantes = df.isnull().sum()
print(valores_faltantes)

# Mostrar también en porcentaje para mejor comprensión
print(f"\nPorcentaje de valores faltantes:")
porcentaje_faltantes = (df.isnull().sum() / len(df)) * 100
for columna, porcentaje in porcentaje_faltantes.items():
    if porcentaje > 0:
        print(f"  {columna}: {porcentaje:.2f}%")
    else:
        print(f"  {columna}: 0.00% (sin valores faltantes)")

# Crear copia del DataFrame para trabajar
df_limpio = df.copy()

# LIMPIEZA ADICIONAL: Limpiar datos numéricos con formato europeo
print(f"\nLIMPIEZA ADICIONAL: Limpiando datos numéricos...")

def clean_numeric(value):
    """Limpia valores numéricos con formato europeo"""
    if isinstance(value, str):
        # Si contiene múltiples puntos, es formato europeo (punto = miles)
        if value.count('.') > 1:
            # Remover todos los puntos excepto el último (separador decimal)
            parts = value.split('.')
            if len(parts) > 2:
                return float(''.join(parts[:-1]) + '.' + parts[-1])
        try:
            return float(value)
        except:
            return value
    return value

# Aplicar limpieza a columnas numéricas
numeric_columns = ['Fare', 'Age', 'Passengerid']
for col in numeric_columns:
    if col in df_limpio.columns:
        print(f"Limpiando columna {col}...")
        df_limpio[col] = df_limpio[col].apply(clean_numeric)

# PUNTO 4: Llena los valores nulos de la columna Age con la mediana
print(f"\nPUNTO 4: Llenando valores nulos de Age con la mediana:")
print("=" * 55)

# Buscar la columna Age (puede tener diferentes nombres)
age_columns = [col for col in df_limpio.columns if 'age' in col.lower()]

if age_columns:
    age_col = age_columns[0]
    valores_nulos_age_antes = df_limpio[age_col].isnull().sum()
    print(f"Columna encontrada: '{age_col}'")
    print(f"Valores nulos en {age_col} antes: {valores_nulos_age_antes}")
    
    if valores_nulos_age_antes > 0:
        mediana_age = df_limpio[age_col].median()
        print(f"Mediana calculada: {mediana_age:.2f}")
        df_limpio[age_col] = df_limpio[age_col].fillna(mediana_age)
        valores_nulos_age_despues = df_limpio[age_col].isnull().sum()
        print(f"Valores nulos en {age_col} después: {valores_nulos_age_despues}")
        print(f"Se llenaron {valores_nulos_age_antes} valores faltantes con la mediana")
    else:
        print(f"La columna {age_col} no tiene valores nulos (ya está limpia)")
        print(f"Mediana de {age_col}: {df_limpio[age_col].median():.2f}")
else:
    print("No se encontró columna Age en el dataset")

# Llenar valores nulos en otras columnas numéricas
print(f"\nLlenando valores nulos en otras columnas numéricas...")
for col in ['Fare']:
    if col in df_limpio.columns:
        nulos_antes = df_limpio[col].isnull().sum()
        if nulos_antes > 0:
            mediana = df_limpio[col].median()
            df_limpio[col] = df_limpio[col].fillna(mediana)
            print(f"  {col}: {nulos_antes} nulos llenados con mediana {mediana:.2f}")

# PUNTO 5: Reemplaza la columna Sex por variables dummy (pd.get_dummies)
print(f"\nPUNTO 5: Reemplazando columna Sex por variables dummy:")
print("=" * 58)

# Buscar la columna Sex
sex_columns = [col for col in df_limpio.columns if 'sex' in col.lower()]

if sex_columns:
    sex_col = sex_columns[0]
    print(f"Columna encontrada: '{sex_col}'")
    print(f"Valores únicos antes: {df_limpio[sex_col].unique()}")
    print(f"Tipo de datos: {df_limpio[sex_col].dtype}")
    
    # FORZAR el uso de pd.get_dummies como requiere el enunciado
    if df_limpio[sex_col].dtype != 'object':
        # Si es numérico, convertir a texto primero para poder usar get_dummies
        print("Convirtiendo valores numéricos a texto para usar pd.get_dummies...")
        df_limpio[sex_col] = df_limpio[sex_col].map({0: 'female', 1: 'male'})
        print(f"Valores después de conversión: {df_limpio[sex_col].unique()}")
    
    # Usar pd.get_dummies como específicamente requiere el enunciado
    print("Aplicando pd.get_dummies...")
    df_limpio = pd.get_dummies(df_limpio, columns=[sex_col], drop_first=True)
    
    # Mostrar las nuevas columnas creadas
    nuevas_columnas_sex = [col for col in df_limpio.columns if sex_col in col]
    print(f"Nuevas columnas creadas: {nuevas_columnas_sex}")
    print(f"Columnas totales después de get_dummies: {list(df_limpio.columns)}")
    
else:
    print("No se encontró columna Sex en el dataset")

# LIMPIEZA FINAL: Asegurar que todas las columnas numéricas sean numéricas
print(f"\nLIMPIEZA FINAL: Verificando tipos de datos...")
for col in df_limpio.columns:
    if col not in ['Embarked']:  # Excluir columnas categóricas conocidas
        try:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors='ignore')
        except:
            pass

# Llenar cualquier valor NaN restante
df_limpio = df_limpio.fillna(df_limpio.median(numeric_only=True))

# Verificar que no hay valores faltantes después de la limpieza
print(f"\nVerificación final de valores faltantes:")
valores_faltantes_final = df_limpio.isnull().sum()
print(valores_faltantes_final)

# PUNTO 6: Guarda el nuevo DataFrame limpio en un archivo titanic_limpio.csv
print(f"\nPUNTO 6: Guardando DataFrame limpio en titanic_limpio.csv:")
print("=" * 62)

try:
    df_limpio.to_csv("titanic_limpio.csv", index=False)
    print(f"Archivo guardado exitosamente: 'titanic_limpio.csv'")
    print(f"Tamaño del archivo: {df_limpio.shape[0]:,} filas × {df_limpio.shape[1]} columnas")
    print(f"Columnas en el archivo limpio:")
    for i, col in enumerate(df_limpio.columns, 1):
        print(f"  {i:2d}. {col}")
        
except Exception as e:
    print(f"Error al guardar el archivo: {e}")

# Mostrar muestra del DataFrame limpio final
print(f"\nMuestra del DataFrame limpio final:")
print("=" * 40)
print(df_limpio.head())

# Verificar tipos de datos finales
print(f"\nTipos de datos finales:")
print(df_limpio.dtypes)

# Resumen de cambios realizados
print(f"\nRESUMEN DE CAMBIOS REALIZADOS:")
print("-" * 35)
print(f"• Dataset original: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"• Dataset limpio: {df_limpio.shape[0]:,} filas × {df_limpio.shape[1]} columnas")
print(f"• Datos numéricos limpiados de formato europeo")
print(f"• Valores nulos completados")
if age_columns and age_col in df.columns:
    age_antes = df[age_col].isnull().sum()
    age_despues = df_limpio[age_col].isnull().sum() if age_col in df_limpio.columns else 0
    print(f"• Valores nulos en Age: {age_antes} → {age_despues}")
if sex_columns:
    print(f"• Columna Sex convertida a variables dummy usando pd.get_dummies")
print(f"• Archivo limpio guardado: titanic_limpio.csv")
print(f"• Dataset listo para machine learning")