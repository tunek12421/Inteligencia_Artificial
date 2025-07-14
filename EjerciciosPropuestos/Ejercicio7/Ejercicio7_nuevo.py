import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
df = pd.read_csv('../Ejercicio1/titanic_limpio.csv')
X = df.drop('2urvived', axis=1)
y = df['2urvived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=42)
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
plt.figure(figsize=(15,10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Sí'], 
filled=True)
plt.show()
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))