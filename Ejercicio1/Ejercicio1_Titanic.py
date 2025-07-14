# ==============================================
# EJERCICIO 1 (FÁCIL): Clasificación binaria con Titanic
# Dataset: Titanic - Machine Learning from Disaster (Kaggle)
# Técnica: Regresión Logística
# ==============================================
# Paso 1: Importar bibliotecas necesarias
import pandas as pd # Para manejo de datos en forma de tablas
import numpy as np # Para operaciones numéricas
from sklearn.model_selection import train_test_split # Para dividir datos
from sklearn.linear_model import LogisticRegression # Modelo de clasificación
from sklearn.metrics import accuracy_score, confusion_matrix # Métricas
import seaborn as sns # Visualizaciones
import matplotlib.pyplot as plt # Gráficos
# Paso 2: Cargar el dataset
# Suponemos que el archivo 'train.csv' está en el mismo directorio que este código
# Si está en otra ruta, reemplazar la ruta por la correcta
df = pd.read_csv("train.csv", sep=';')

# Limpiar el dataset - eliminar columnas 'zero' y renombrar
important_cols = [col for col in df.columns if 'zero' not in col.lower()]
df = df[important_cols].copy()
df = df.rename(columns={'Passengerid': 'PassengerId', '2urvived': 'Survived'})

print("Vista previa de los datos:")
print(df.head())
# Paso 3: Seleccionar variables útiles para el modelo
# Usaremos: 'Age', 'Fare', 'Sex' como características y 'Survived' como etiqueta
data = df[['PassengerId', 'Age', 'Fare', 'Sex', 'Survived']].copy()

# Limpiar valores de Fare que tienen formato incorrecto (punto como separador de miles)
def clean_fare(value):
    if isinstance(value, str):
        # Si contiene múltiples puntos, es formato europeo (punto = miles)
        if value.count('.') > 1:
            # Remover todos los puntos excepto el último (separador decimal)
            parts = value.split('.')
            if len(parts) > 2:
                return float(''.join(parts[:-1]) + '.' + parts[-1])
        return float(value)
    return value

data['Fare'] = data['Fare'].apply(clean_fare)

# Paso 4: Preprocesamiento de los datos
# 4.1: Llenar valores faltantes en la columna 'Age' con la mediana
data = data.copy()  # Evitar warning de pandas
data['Age'] = data['Age'].fillna(data['Age'].median())

# 4.2: Convertir variable categórica 'Sex' en variable numérica
# Sex ya es numérica (0=female, 1=male), pero creamos columna Sex_male para consistencia
data['Sex_male'] = data['Sex']
data = data.drop(['Sex', 'PassengerId'], axis=1)

# Paso 5: Separar características (X) y etiqueta (y)
X = data.drop('Survived', axis=1) # Variables de entrada
y = data['Survived'] # Variable objetivo
# Paso 6: Dividir en datos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Paso 7: Crear y entrenar el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000) # Aumentamos el número máximo de iteraciones
model.fit(X_train, y_train)
# Paso 8: Realizar predicciones
predictions = model.predict(X_test)
# Paso 9: Evaluar el modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Precisión del modelo: {accuracy:.2f}")
# Matriz de confusión para entender errores
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title("Matriz de confusión")
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta real")
plt.show()
# Explicación:
# - La precisión indica qué porcentaje de las predicciones fueron correctas.
# - La matriz de confusión muestra cómo se distribuyen los aciertos y errores:
# [ [TN, FP],
# [FN, TP] ]
# TN: verdaderos negativos, FP: falsos positivos,
# FN: falsos negativos, TP: verdaderos positivos