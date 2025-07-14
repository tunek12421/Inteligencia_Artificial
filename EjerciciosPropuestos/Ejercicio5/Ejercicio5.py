# ==============================================
# EJERCICIO 5: Clasificación de texto con TF-IDF - Noticias falsas vs verdaderas
# Dataset: noticias_procesadas.csv (generado en Ejercicio 3)
# Técnica: TF-IDF + Naive Bayes para clasificación de texto
# ==============================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("=== EJERCICIO 5: CLASIFICACIÓN DE TEXTO CON TF-IDF - FAKE NEWS ===\n")

# Paso 1: Carga de datos
print("Paso 1: Cargando datos de noticias procesadas...")
try:
    df = pd.read_csv('../Ejercicio3/noticias_procesadas.csv')
    print(f"Dataset cargado: {df.shape[0]:,} noticias x {df.shape[1]} columnas")
    print(f"Columnas disponibles: {list(df.columns)}")
except FileNotFoundError:
    print("Error: No se encontró el archivo noticias_procesadas.csv")
    print("Ejecuta primero el Ejercicio 3 para generar el archivo")
    exit()

print("\nVista previa del dataset:")
print(df.head())

# Verificar distribución de clases
print(f"\nDistribución de etiquetas:")
distribucion = df['label'].value_counts().sort_index()
for valor, cantidad in distribucion.items():
    tipo = "Noticias falsas" if valor == 0 else "Noticias verdaderas"
    porcentaje = (cantidad / len(df)) * 100
    print(f"  {tipo} ({valor}): {cantidad:,} noticias ({porcentaje:.1f}%)")

# Estadísticas del texto
print(f"\nEstadísticas del texto:")
longitudes = df['text'].str.len()
print(f"Longitud promedio: {longitudes.mean():.0f} caracteres")
print(f"Longitud mínima: {longitudes.min()} caracteres")
print(f"Longitud máxima: {longitudes.max():,} caracteres")
print(f"Mediana: {longitudes.median():.0f} caracteres")

# Paso 2: Vectorización con TF-IDF
print(f"\nPaso 2: Vectorizando texto con TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words='english',    # Eliminar palabras comunes en inglés
    max_features=5000,       # Máximo 5000 características más importantes
    min_df=2,               # Palabra debe aparecer al menos en 2 documentos
    max_df=0.95,            # Palabra no debe aparecer en más del 95% de documentos
    ngram_range=(1, 2)      # Usar unigramas y bigramas
)

X = vectorizer.fit_transform(df['text'])
y = df['label']

print(f"Matriz TF-IDF creada:")
print(f"  Forma: {X.shape} (noticias x características)")
print(f"  Tipo: {type(X)} (matriz dispersa)")
print(f"  Características extraídas: {len(vectorizer.get_feature_names_out()):,}")

# Mostrar algunas características importantes
feature_names = vectorizer.get_feature_names_out()
print(f"\nEjemplos de características (palabras/n-gramas):")
for i in range(min(20, len(feature_names))):
    print(f"  {i+1:2d}. '{feature_names[i]}'")

# Paso 3: División de datos
print(f"\nPaso 3: Dividiendo datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Datos de entrenamiento: {X_train.shape[0]:,} noticias")
print(f"Datos de prueba: {X_test.shape[0]:,} noticias")
print(f"Proporción de datos de prueba: 30%")

# Verificar distribución en train/test
print(f"\nDistribución en conjunto de entrenamiento:")
train_dist = pd.Series(y_train).value_counts().sort_index()
for valor, cantidad in train_dist.items():
    tipo = "Falsas" if valor == 0 else "Verdaderas"
    porcentaje = (cantidad / len(y_train)) * 100
    print(f"  {tipo}: {cantidad:,} ({porcentaje:.1f}%)")

# Paso 4: Entrenamiento con Naive Bayes
print(f"\nPaso 4: Entrenando modelo Naive Bayes...")
model = MultinomialNB(alpha=1.0)  # alpha es el parámetro de suavizado
model.fit(X_train, y_train)
print("Modelo Multinomial Naive Bayes entrenado exitosamente")

# Paso 5: Realizar predicciones
print(f"\nPaso 5: Realizando predicciones...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Paso 6: Evaluación del modelo
print(f"\nPaso 6: Evaluando el modelo...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\nReporte de clasificación detallado:")
print(classification_report(y_test, y_pred, target_names=['Noticia falsa', 'Noticia verdadera']))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Falsa', 'Verdadera'], 
            yticklabels=['Falsa', 'Verdadera'])
plt.title('Matriz de confusión - Clasificación de Fake News')
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.tight_layout()
plt.savefig('confusion_matrix_fake_news.png', dpi=300, bbox_inches='tight')
print("Matriz de confusión guardada como 'confusion_matrix_fake_news.png'")
plt.close()

# Análisis de palabras más importantes
print(f"\nAnálisis de palabras más importantes para cada clase:")

# Obtener probabilidades logarítmicas de las características
log_probs = model.feature_log_prob_
feature_names = vectorizer.get_feature_names_out()

# Top palabras para noticias falsas (clase 0)
top_fake_indices = np.argsort(log_probs[0])[-10:][::-1]
print(f"\nTop 10 palabras indicativas de NOTICIAS FALSAS:")
for i, idx in enumerate(top_fake_indices, 1):
    word = feature_names[idx]
    score = log_probs[0][idx]
    print(f"  {i:2d}. '{word}' (score: {score:.3f})")

# Top palabras para noticias verdaderas (clase 1)
top_true_indices = np.argsort(log_probs[1])[-10:][::-1]
print(f"\nTop 10 palabras indicativas de NOTICIAS VERDADERAS:")
for i, idx in enumerate(top_true_indices, 1):
    word = feature_names[idx]
    score = log_probs[1][idx]
    print(f"  {i:2d}. '{word}' (score: {score:.3f})")

# Ejemplos de predicciones
print(f"\nEjemplos de predicciones:")
print("Real      | Predicho  | Probabilidad | Texto (primeros 80 caracteres)")
print("-" * 80)
for i in range(min(5, len(y_test))):
    real = "Falsa" if y_test.iloc[i] == 0 else "Verdadera"
    pred = "Falsa" if y_pred[i] == 0 else "Verdadera"
    prob = y_pred_proba[i][1]  # Probabilidad de ser verdadera
    texto = df.iloc[y_test.index[i]]['text'][:80] + "..."
    print(f"{real:9} | {pred:9} | {prob:.3f}        | {texto}")

# Análisis detallado de la matriz de confusión
tn, fp, fn, tp = conf_matrix.ravel()
print(f"\nAnálisis detallado de la matriz de confusión:")
print(f"Verdaderos Negativos (TN): {tn:,} - Predijo 'Falsa' y era correcto")
print(f"Falsos Positivos (FP): {fp:,} - Predijo 'Verdadera' pero era falsa")
print(f"Falsos Negativos (FN): {fn:,} - Predijo 'Falsa' pero era verdadera")
print(f"Verdaderos Positivos (TP): {tp:,} - Predijo 'Verdadera' y era correcto")

# Métricas adicionales
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMétricas adicionales:")
print(f"Precisión: {precision:.4f} - De las noticias que predijo como verdaderas, cuántas eran correctas")
print(f"Recall: {recall:.4f} - De las noticias verdaderas reales, cuántas detectó")
print(f"F1-Score: {f1_score:.4f} - Media armónica de precisión y recall")

# Estadísticas finales
print(f"\nESTADÍSTICAS FINALES:")
print("=" * 30)
print(f"Total de noticias analizadas: {len(y_test):,}")
print(f"Predicciones correctas: {(y_test == y_pred).sum():,}")
print(f"Predicciones incorrectas: {(y_test != y_pred).sum():,}")
print(f"Precisión final: {accuracy:.4f}")

print(f"\nCaracterísticas del modelo TF-IDF:")
print(f"Vocabulario total: {len(feature_names):,} términos")
print(f"Matriz de características: {X.shape[0]:,} x {X.shape[1]:,}")
print(f"Densidad de la matriz: {X.nnz / (X.shape[0] * X.shape[1]):.6f}")

print(f"\nModelo de clasificación de texto completado exitosamente")
print(f"Archivo generado: confusion_matrix_fake_news.png")