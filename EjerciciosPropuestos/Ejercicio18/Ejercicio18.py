import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
# Cargar el dataset de Titanic
df = pd.read_csv('../Ejercicio1/titanic_limpio.csv')
X = df.drop('2urvived', axis=1)
y = df['2urvived']
# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar modelos
models = {
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(),
    "CatBoost": CatBoostClassifier(silent=True)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} - Precisión: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))