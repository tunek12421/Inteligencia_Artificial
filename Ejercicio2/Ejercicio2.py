# ==============================================
# EJERCICIO 2 (FÁCIL): Clasificación multiclase con el dataset Iris
# Dataset: Iris dataset incluido en sklearn
# Técnica: K-Nearest Neighbors (KNN)
# ==============================================
# Paso 1: Importar bibliotecas necesarias
from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Paso 2: Cargar el dataset
iris = load_iris()
X = iris.data # Características (4 variables por flor)
y = iris.target # Etiqueta (0: setosa, 1: versicolor, 2: virginica)
# Mostrar información general del dataset
print("Nombre de las clases:", iris.target_names)
print("Nombres de las características:", iris.feature_names)
# Paso 3: Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Paso 4: Crear el modelo KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Paso 5: Realizar predicciones
predictions = knn.predict(X_test)
# Paso 6: Evaluar el modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Precisión del modelo KNN (k=3): {accuracy:.2f}")
# Paso 7: Visualización de matriz de confusión
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, cmap='Greens', fmt='d')
plt.title("Matriz de confusión - KNN")
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta real")
#plt.show()
plt.savefig('confusion_matrix_iris.png', dpi=300, bbox_inches='tight')
print("Matriz de confusión guardada como 'confusion_matrix_iris.png'")
# plt.show()  # Descomenta si tienes interfaz gráfica
# Explicación:
# - KNN clasifica observaciones nuevas basándose en las más cercanas (vecinos).
# - En este ejemplo, k=3 significa que se miran las 3 observaciones más cercanas para
#decidir la clase.
# - Es un algoritmo intuitivo pero sensible al escalado y a los valores de k.
