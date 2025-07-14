# ==============================================
# EJERCICIO 1-4: Limpieza de datos para Machine Learning - Titanic
# Dataset: Titanic - train.csv from Kaggle
# Objetivo: Limpieza completa para uso en modelos de ML (Ejercicio 4)
# ==============================================

import pandas as pd
import numpy as np

print("=== EJERCICIO 1-4: LIMPIEZA PARA MACHINE LEARNING - TITANIC ===\n")

# PUNTO 1: Carga el archivo CSV con pandas
try:
    df = pd.read_csv("../../Dataset/train.csv", sep=';')
    print("PUNTO 1: Archivo train.csv cargado exitosamente con pandas")
    
    if df.shape[1] == 1:
        df = pd.read_csv("../../Dataset/train.csv", sep=',')
        
except FileNotFoundError:
    print("Error: No se encontró el archivo train.csv")
    print("Asegúrate de que el archivo esté en la carpeta Dataset/")
    exit()

# Limpiar el dataset - eliminar columnas 'zero' si existen
important_cols = [col for col in df.columns if 'zero' not in col.lower()]
df = df[important_cols].copy()

print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"Columnas disponibles: {list(df.columns)}")

# PUNTO 2: Muestra las primeras 10 filas con .head()
print(f"\nPUNTO 2: Primeras 10 filas del dataset:")
print("=" * 60)
print(df.head(10))

# PUNTO 3: Calcula cuántos valores faltantes hay por columna
print(f"\nPUNTO 3: Valores faltantes por columna:")
print("=" * 45)
valores_faltantes = df.isnull().sum()
print(valores_faltantes)

print(f"\nPorcentaje de valores faltantes:")
porcentaje_faltantes = (df.isnull().sum() / len(df)) * 100
for columna, porcentaje in porcentaje_faltantes.items():
    if porcentaje > 0:
        print(f"  {columna}: {porcentaje:.2f}%")
    else:
        print(f"  {columna}: 0.00% (sin valores faltantes)")

# Crear copia del DataFrame para trabajar
df_limpio = df.copy()

# LIMPIEZA COMPLETA PARA MACHINE LEARNING
print(f"\nLIMPIEZA COMPLETA PARA MACHINE LEARNING:")
print("=" * 45)

# Función para limpiar valores en formato europeo
def clean_numeric_value(value):
    if isinstance(value, str):
        if value.count('.') > 1:
            parts = value.split('.')
            if len(parts) > 2:
                return float(''.join(parts[:-1]) + '.' + parts[-1])
        try:
            return float(value)
        except:
            return np.nan
    return value

# Aplicar limpieza a TODAS las columnas numéricas potenciales
numeric_potential_cols = ['Passengerid', 'Age', 'Fare', 'sibsp', 'Parch', 'Pclass', 'Embarked']
for col in numeric_potential_cols:
    if col in df_limpio.columns:
        print(f"Limpiando columna '{col}'...")
        df_limpio[col] = df_limpio[col].apply(clean_numeric_value)

# PUNTO 4: Llena los valores nulos de la columna Age con la mediana
print(f"\nPUNTO 4: Llenando valores nulos de Age con la mediana:")
print("=" * 55)

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

# Llenar otros valores faltantes numéricos
print(f"\nLlenando otros valores faltantes:")
for col in ['Fare', 'sibsp', 'Parch', 'Pclass', 'Embarked']:
    if col in df_limpio.columns:
        nulos_antes = df_limpio[col].isnull().sum()
        if nulos_antes > 0:
            if col in ['sibsp', 'Parch']:
                df_limpio[col] = df_limpio[col].fillna(0)
                print(f"  {col}: {nulos_antes} valores llenados con 0")
            else:
                mediana = df_limpio[col].median()
                df_limpio[col] = df_limpio[col].fillna(mediana)
                print(f"  {col}: {nulos_antes} valores llenados con mediana ({mediana:.2f})")

# PUNTO 5: Reemplaza la columna Sex por variables dummy (pd.get_dummies)
print(f"\nPUNTO 5: Reemplazando columna Sex por variables dummy:")
print("=" * 58)

sex_columns = [col for col in df_limpio.columns if 'sex' in col.lower()]

if sex_columns:
    sex_col = sex_columns[0]
    print(f"Columna encontrada: '{sex_col}'")
    print(f"Valores únicos antes: {df_limpio[sex_col].unique()}")
    print(f"Tipo de datos: {df_limpio[sex_col].dtype}")
    
    if df_limpio[sex_col].dtype != 'object':
        print("Convirtiendo valores numéricos a texto para usar pd.get_dummies...")
        df_limpio[sex_col] = df_limpio[sex_col].map({0: 'female', 1: 'male'})
        print(f"Valores después de conversión: {df_limpio[sex_col].unique()}")
    
    print("Aplicando pd.get_dummies...")
    df_limpio = pd.get_dummies(df_limpio, columns=[sex_col], drop_first=True)
    
    nuevas_columnas_sex = [col for col in df_limpio.columns if sex_col in col]
    print(f"Nuevas columnas creadas: {nuevas_columnas_sex}")
    print(f"Columnas totales después de get_dummies: {list(df_limpio.columns)}")
    
else:
    print("No se encontró columna Sex en el dataset")

# CONVERSIÓN COMPLETA A TIPOS NUMÉRICOS PARA ML
print(f"\nCONVERSIÓN A TIPOS NUMÉRICOS PARA MACHINE LEARNING:")
print("=" * 55)

# Convertir Sex_male de boolean a int (obligatorio para sklearn)
if 'Sex_male' in df_limpio.columns:
    print(f"Convirtiendo 'Sex_male' de {df_limpio['Sex_male'].dtype} a int...")
    df_limpio['Sex_male'] = df_limpio['Sex_male'].astype(int)
    print(f"Conversión completada: {df_limpio['Sex_male'].dtype}")

# Asegurar que todas las columnas numéricas sean del tipo correcto
for col in df_limpio.columns:
    if col != 'Sex_male':  # Ya convertida arriba
        try:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
        except:
            pass

print(f"\nTipos de datos finales:")
print(df_limpio.dtypes)

# Verificar que no hay valores faltantes después de la limpieza
print(f"\nVerificación final de valores faltantes:")
valores_faltantes_final = df_limpio.isnull().sum()
print(valores_faltantes_final)

if valores_faltantes_final.sum() > 0:
    print(f"ADVERTENCIA: Aún hay {valores_faltantes_final.sum()} valores faltantes")
    print("Eliminando filas con valores faltantes...")
    df_limpio = df_limpio.dropna()
    print(f"Filas restantes después de eliminar NaN: {len(df_limpio)}")

# PUNTO 6: Guarda el nuevo DataFrame limpio
print(f"\nPUNTO 6: Guardando DataFrame limpio para ML:")
print("=" * 45)

try:
    df_limpio.to_csv("titanic_limpio_ml.csv", index=False)
    print(f"Archivo guardado exitosamente: 'titanic_limpio_ml.csv'")
    print(f"Tamaño del archivo: {df_limpio.shape[0]:,} filas x {df_limpio.shape[1]} columnas")
    print(f"Columnas en el archivo limpio:")
    for i, col in enumerate(df_limpio.columns, 1):
        print(f"  {i:2d}. {col} ({df_limpio[col].dtype})")
        
except Exception as e:
    print(f"Error al guardar el archivo: {e}")

# Verificación de compatibilidad con sklearn
print(f"\nVERIFICACIÓN DE COMPATIBILIDAD CON SKLEARN:")
print("=" * 50)

sklearn_compatible = True
for col in df_limpio.columns:
    if df_limpio[col].dtype == 'object':
        print(f"PROBLEMA: Columna '{col}' es de tipo object")
        sklearn_compatible = False
    elif df_limpio[col].dtype == 'bool':
        print(f"PROBLEMA: Columna '{col}' es de tipo bool")
        sklearn_compatible = False

if sklearn_compatible:
    print("PERFECTO: Todas las columnas son compatibles con sklearn")
    print("Dataset listo para modelos de Machine Learning")
else:
    print("ERROR: Dataset NO compatible con sklearn")

# Mostrar muestra del DataFrame limpio final
print(f"\nMuestra del DataFrame limpio final:")
print("=" * 40)
print(df_limpio.head())

# RESUMEN FINAL
print(f"\nRESUMEN FINAL:")
print("-" * 20)
print(f"• Dataset original: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"• Dataset limpio ML: {df_limpio.shape[0]:,} filas x {df_limpio.shape[1]} columnas")
print(f"• Todas las columnas numéricas: SÍ")
print(f"• Sin valores faltantes: {'SÍ' if df_limpio.isnull().sum().sum() == 0 else 'NO'}")
print(f"• Compatible con sklearn: {'SÍ' if sklearn_compatible else 'NO'}")
print(f"• Archivo generado: titanic_limpio_ml.csv")

print(f"\nEJERCICIO 1-4 COMPLETADO - DATASET LISTO PARA ML!")