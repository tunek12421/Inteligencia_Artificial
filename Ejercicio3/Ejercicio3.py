# ==============================================
# EJERCICIO 3 (FÁCIL): Regresión lineal simple con datos sintéticos
# Dataset: Generado manualmente
# Técnica: Regresión lineal con una sola variable (una característica)
# ==============================================
# Paso 1: Importar bibliotecas necesarias
from sklearn.linear_model import LinearRegression # Modelo de regresión lineal


import matplotlib.pyplot as plt
import numpy as np
# Paso 2: Crear datos de ejemplo (población vs ganancia)
poblacion = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 6.5]) # en millones
ganancia = np.array([1.0, 1.5, 2.0, 2.5, 2.7, 3.5, 3.8, 4.2, 5.0, 5.5]) # en cientos de miles de dólares
# Paso 3: Redimensionar datos (para que sklearn los acepte)
X = poblacion.reshape(-1, 1) # X debe ser una matriz 2D
y = ganancia # y puede ser un vector 1D
# Paso 4: Crear el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y) # Entrenamos el modelo con los datos
# Paso 5: Visualizar los datos y la recta de regresión
plt.scatter(poblacion, ganancia, color='blue', label='Datos reales')
plt.plot(poblacion, model.predict(X), color='red', label='Línea de regresión')
plt.title("Regresión lineal: Población vs Ganancia")
plt.xlabel("Población (millones)")
plt.ylabel("Ganancia (cientos de miles de $)")
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('confusion_matrix_regresionLinear.png', dpi=300, bbox_inches='tight')
print("Matriz de confusión guardada como 'confusion_matrix_regresionLinear.png'")
# Paso 6: Interpretar los coeficientes
print(f"Pendiente (coeficiente): {model.coef_[0]:.2f}")
print(f"Intersección (ordenada al origen): {model.intercept_:.2f}")
# Explicación:
# - La pendiente indica cuánto cambia la ganancia por cada millón adicional de personas.
# - La intersección representa la ganancia estimada si la población fuera 0.
# - La regresión lineal intenta ajustar la mejor línea recta que minimiza el error cuadrático entre los puntos reales y la línea predicha.