from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target
# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=42)
# Entrenar modelos
models = {
 "Logistic Regression": LogisticRegression(max_iter=200),
 "SVM": SVC(),
 "KNN": KNeighborsClassifier()
}
# Evaluar modelos
for name, model in models.items():
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 print(f"--- {name} ---")
 print(classification_report(y_test, y_pred, target_names=iris.target_names))