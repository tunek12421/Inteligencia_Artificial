from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
# Cargar datos
df = pd.read_csv('../Ejercicio1/titanic_limpio.csv')
X = df.drop('2urvived', axis=1)
y = df['2urvived']
# Ajustar con GridSearchCV
param_grid = {'n_estimators': [50, 100], 'max_depth': [4, 6, 8]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X, y)
# Mostrar mejor modelo
print("Mejores parámetros:", grid.best_params_)
print("Reporte de clasificación:", classification_report(y, grid.predict(X)))