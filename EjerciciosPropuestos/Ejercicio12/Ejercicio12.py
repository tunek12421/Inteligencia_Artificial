# ==============================================
# EJERCICIO 12: Detección de anomalías en salarios – Dataset sintético
# Objetivo: Usar Isolation Forest para detectar anomalías en salarios
# ==============================================

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

print("=== EJERCICIO 12: DETECCIÓN DE ANOMALÍAS - SALARIOS ===\n")

# Cargar datos
df = pd.read_csv('../../Dataset/Employee_Salary_Dataset.csv')
print(f"Dataset cargado: {df.shape[0]} empleados")
print("\nPrimeras filas:")
print(df.head())

print("\nEstadísticas descriptivas:")
print(df.describe())

# Preparar datos para detección de anomalías
features = ['Experience_Years', 'Age', 'Salary']
X = df[features]

print(f"\nUsando características: {features}")

# Modelo de detección de anomalías
print("\nEntrenando Isolation Forest...")
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(X)

# Contar anomalías
normales = (df['anomaly'] == 1).sum()
anomalas = (df['anomaly'] == -1).sum()
print(f"Empleados normales: {normales}")
print(f"Empleados anómalos: {anomalas}")

# Visualización
print("\nGenerando visualización...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(data=df, x='Experience_Years', y='Salary', hue='anomaly', palette='Set2')
plt.title('Experiencia vs Salario')

plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='Age', y='Salary', hue='anomaly', palette='Set2')
plt.title('Edad vs Salario')

plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Experience_Years', y='Age', hue='anomaly', palette='Set2')
plt.title('Experiencia vs Edad')

plt.tight_layout()
plt.savefig('anomalias_salarios.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'anomalias_salarios.png'")

# Mostrar empleados anómalos
print("\nEMPLEADOS CON SALARIOS ANÓMALOS:")
print("=" * 40)
anomalos = df[df['anomaly'] == -1][['Experience_Years', 'Age', 'Gender', 'Salary']].sort_values('Salary')
print(anomalos)

print(f"\nRango de salarios anómalos: ${anomalos['Salary'].min():,} - ${anomalos['Salary'].max():,}")
print(f"Salario promedio general: ${df['Salary'].mean():,.0f}")
print(f"Salario promedio anómalos: ${anomalos['Salary'].mean():,.0f}")

print("\nEJERCICIO 12 COMPLETADO")