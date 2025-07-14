import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
df = pd.read_csv('../Ejercicio1/titanic_limpio.csv')
X = df.drop('2urvived', axis=1)
y = df['2urvived']
model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
print("\nÁrbol de decisión exportado como texto")