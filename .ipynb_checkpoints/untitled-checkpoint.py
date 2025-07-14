# %% [markdown]
# # EJERCICIO 1: Clasificación binaria con Titanic
# **Dataset:** Titanic - Machine Learning from Disaster (Kaggle)  
# **Técnica:** Regresión Logística

# %% [markdown]
# ## Paso 1: Importar bibliotecas necesarias

# %%
import pandas as pd # Para manejo de datos en forma de tablas
import numpy as np # Para operaciones numéricas
from sklearn.model_selection import train_test_split # Para dividir datos
from sklearn.linear_model import LogisticRegression # Modelo de clasificación
from sklearn.metrics import accuracy_score, confusion_matrix # Métricas
import seaborn as sns # Visualizaciones
import matplotlib.pyplot as plt # Gráficos

# Configuración para mostrar gráficos en Jupyter
%matplotlib inline
plt.style.use('default')
sns.set_palette("husl")

# %% [markdown]
# ## Paso 2: Cargar el dataset

# %%
# Cargar el CSV con el separador correcto
df = pd.read_csv("train.csv", sep=';')

# Limpiar el dataset - eliminar columnas 'zero' y renombrar
important_cols = [col for col in df.columns if 'zero' not in col.lower()]
df = df[important_cols].copy()
df = df.rename(columns={'Passengerid': 'PassengerId', '2urvived': 'Survived'})

print("Vista previa de los datos:")
print(df.head())
print(f"\nForma del dataset: {df.shape}")

# %% [markdown]
# ## Paso 3: Seleccionar variables útiles para el modelo

# %%
# Usaremos: 'Age', 'Fare', 'Sex' como características y 'Survived' como etiqueta
data = df[['PassengerId', 'Age', 'Fare', 'Sex', 'Survived']].copy()

# Limpiar valores de Fare que tienen formato incorrecto
def clean_fare(value):
    if isinstance(value, str):
        if value.count('.') > 1:
            parts = value.split('.')
            if len(parts) > 2:
                return float(''.join(parts[:-1]) + '.' + parts[-1])
        return float(value)
    return value

data['Fare'] = data['Fare'].apply(clean_fare)

print("Datos seleccionados:")
print(data.head())
print(f"\nInformación del dataset:")
print(data.info())

# %% [markdown]
# ## Paso 4: Preprocesamiento de los datos

# %%
# 4.1: Llenar valores faltantes en la columna 'Age' con la mediana
data = data.copy()  # Evitar warning de pandas
data['Age'] = data['Age'].fillna(data['Age'].median())

# 4.2: Convertir variable categórica 'Sex' en variable numérica
# Sex ya es numérica (0=female, 1=male), pero creamos columna Sex_male para consistencia
data['Sex_male'] = data['Sex']
data = data.drop(['Sex', 'PassengerId'], axis=1)

print("Datos después del preprocesamiento:")
print(data.head())
print(f"\nEstadísticas descriptivas:")
print(data.describe())

# %% [markdown]
# ## Paso 5: Análisis exploratorio de datos

# %%
# Visualizar la distribución de supervivientes
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
data['Survived'].value_counts().plot(kind='bar')
plt.title('Distribución de Supervivientes')
plt.xlabel('Sobrevivió (0=No, 1=Sí)')
plt.ylabel('Cantidad')

plt.subplot(1, 3, 2)
data.groupby('Sex_male')['Survived'].mean().plot(kind='bar')
plt.title('Tasa de Supervivencia por Sexo')
plt.xlabel('Sexo (0=Mujer, 1=Hombre)')
plt.ylabel('Tasa de Supervivencia')

plt.subplot(1, 3, 3)
plt.scatter(data['Age'], data['Fare'], c=data['Survived'], alpha=0.6)
plt.xlabel('Edad')
plt.ylabel('Tarifa')
plt.title('Edad vs Tarifa (Color = Supervivencia)')
plt.colorbar()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Paso 6: Separar características y dividir datos

# %%
# Separar características (X) y etiqueta (y)
X = data.drop('Survived', axis=1) # Variables de entrada
y = data['Survived'] # Variable objetivo

print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")
print(f"Características: {list(X.columns)}")

# Dividir en datos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDivisión de datos:")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# %% [markdown]
# ## Paso 7: Crear y entrenar el modelo

# %%
# Crear y entrenar el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000) # Aumentamos el número máximo de iteraciones
model.fit(X_train, y_train)

print("✅ Modelo entrenado exitosamente!")

# %% [markdown]
# ## Paso 8: Realizar predicciones y evaluar el modelo

# %%
# Realizar predicciones
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluar el modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Precisión del modelo: {accuracy:.2%}")

# Mostrar ejemplos de predicciones
results_df = pd.DataFrame({
    'Real': y_test.iloc[:10].values,
    'Predicción': predictions[:10],
    'Probabilidad': probabilities[:10]
})
print(f"\nPrimeras 10 predicciones:")
print(results_df)

# %% [markdown]
# ## Paso 9: Visualizar resultados

# %%
# Matriz de confusión
conf_matrix = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=['No sobrevivió', 'Sobrevivió'],
            yticklabels=['No sobrevivió', 'Sobrevivió'])
plt.title("Matriz de confusión")
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta real")
plt.show()

# Importancia de características (coeficientes)
feature_importance = pd.DataFrame({
    'Característica': X.columns,
    'Coeficiente': model.coef_[0]
})

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Característica'], feature_importance['Coeficiente'])
plt.title('Coeficientes del Modelo de Regresión Logística')
plt.xlabel('Coeficiente')
plt.grid(True, alpha=0.3)
plt.show()

print("Importancia de características:")
print(feature_importance.sort_values('Coeficiente', key=abs, ascending=False))

# %% [markdown]
# ## Explicación de resultados:
# 
# - **La precisión** indica qué porcentaje de las predicciones fueron correctas.
# - **La matriz de confusión** muestra cómo se distribuyen los aciertos y errores:
#   - **TN (Verdaderos Negativos)**: Predicho no sobrevive, real no sobrevive
#   - **FP (Falsos Positivos)**: Predicho sobrevive, real no sobrevive  
#   - **FN (Falsos Negativos)**: Predicho no sobrevive, real sobrevive
#   - **TP (Verdaderos Positivos)**: Predicho sobrevive, real sobrevive
# - **Los coeficientes** muestran qué tan importante es cada característica:
#   - Coeficiente positivo: aumenta la probabilidad de supervivencia
#   - Coeficiente negativo: disminuye la probabilidad de supervivencia