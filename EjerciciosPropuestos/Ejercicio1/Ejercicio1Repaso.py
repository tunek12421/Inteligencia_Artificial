import pandas as pd
import numpy as np

try:
    df = pd.read_csv("../../Dataset/train.csv", sep=';')
    print("\n\nPunto 1: Archivo cargado con pandas\n")

    if df.shape[1] == 1:
        df = pd.read_csv("../../Dataset/train.csv", sep=',')

except FileNotFoundError:
    print("Error: no se encontro el archivo train.csv")
    exit()

print(f"Dataser cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"Columnas disponibles: {list(df.columns)}")


print(f"\nPunto 2: Primeras 10 filas del dataset:")
print("=" * 30)
print(df.head(10))

print(f"\nPunto 3: Valores faltantes por columna:")
print("=" * 30)

valores_faltantes = df.isnull().sum()
print(valores_faltantes)
