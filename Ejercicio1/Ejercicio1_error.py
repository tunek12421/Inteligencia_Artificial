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
df = pd.read_csv("../train.csv", sep=';')

# Limpiar el dataset - eliminar columnas 'zero' y renombrar
important_cols = [col for col in df.columns if 'zero' not in col.lower()]
df = df[important_cols].copy()

print("Vista previa de los datos:")
print(df.head())
# Paso 3: Seleccionar variables útiles para el modelo
# Usaremos: 'Pclass', 'Sex', 'Age', 'Fare' como características y 'Survived' como etiqueta
data = df[['Passengerid', 'Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked', '2urvived']].copy()

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
# 4.1: Verificar valores faltantes
print(f"\nValores faltantes antes de limpieza:")
print(data.isnull().sum())

# 4.2: Llenar valores faltantes en todas las columnas
data = data.copy()  # Evitar warning de pandas

# Llenar Age con la mediana
data['Age'] = data['Age'].fillna(data['Age'].median())

# Llenar Fare con la mediana  
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Llenar Embarked con la moda (valor más frecuente)
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Llenar sibsp y Parch con 0 (valor más lógico)
data['sibsp'] = data['sibsp'].fillna(0)
data['Parch'] = data['Parch'].fillna(0)

# Llenar Pclass con la moda
data['Pclass'] = data['Pclass'].fillna(data['Pclass'].mode()[0])

print(f"\nValores faltantes después de limpieza:")
print(data.isnull().sum())

# 4.3: Convertir variable categórica 'Sex' en variable numérica
# Como Sex ya es numérico (0, 1), no necesitamos get_dummies, pero lo mantenemos para consistencia
if data['Sex'].dtype == 'object':
    data = pd.get_dummies(data, columns=['Sex'], drop_first=True) # 'Sex_male' será 1 si es hombre, 0 si mujer
else:
    # Sex ya es numérico, crear columna Sex_male para consistencia
    data['Sex_male'] = data['Sex']
    data = data.drop('Sex', axis=1)

# Paso 5: Separar características (X) y etiqueta (y)
X = data.drop('2urvived', axis=1) # Variables de entrada
y = data['2urvived'] # Variable objetivo
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
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title("Matriz de confusión")
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta real")
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Matriz de confusión guardada como 'confusion_matrix.png'")
# plt.show()  # Descomenta si tienes interfaz gráfica
# Explicación:
# - La precisión indica qué porcentaje de las predicciones fueron correctas.
# - La matriz de confusión muestra cómo se distribuyen los aciertos y errores:
# [ [TN, FP],
# [FN, TP] ]
# TN: verdaderos negativos, FP: falsos positivos,
# FN: falsos negativos, TP: verdaderos positivos