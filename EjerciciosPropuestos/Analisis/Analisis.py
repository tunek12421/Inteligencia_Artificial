# ==============================================
# ANÁLISIS FINAL DE RESULTADOS
# ==============================================

print("ANÁLISIS FINAL DE RESULTADOS")
print()
print("EJERCICIO 4: CLASIFICACIÓN BINARIA - TITANIC")
print()
print("¿Qué variables influyeron más en el modelo?")
print("Principalmente el sexo. Los hombres tenían muchas menos probabilidades de sobrevivir (+2.312 para Sex_male). También la clase social importó bastante, los de primera clase se salvaron más (-0.604 para Pclass). Si tenías hermanos o esposa a bordo también empeoraba tus chances (-0.199 para sibsp). Curiosamente el precio del boleto no afectó casi nada.")
print()
print("¿Qué métricas fueron mejores y por qué?")
print("El modelo tuvo 86.51% de precisión general. Fue muy bueno prediciendo quién iba a morir (96% de recall) pero malo prediciendo supervivientes (solo 61%). Esto pasa porque murió mucha más gente (73.9%) que la que sobrevivió (26.1%), entonces el modelo aprendió a ser pesimista.")
print()
print("¿Cómo mejorarías este modelo para un caso real?")
print("Haría variables nuevas combinando datos, como el tamaño total de la familia o extraer títulos de los nombres. Cambiaría a Random Forest que maneja mejor las interacciones. El dataset está muy desbalanceado así que usaría SMOTE para crear supervivientes sintéticos. También validación cruzada para estar más seguro de los resultados.")
print()
print("EJERCICIO 5: CLASIFICACIÓN DE TEXTO - FAKE NEWS")
print()
print("¿Qué variables influyeron más en el modelo?")
print("Las noticias falsas mencionaban mucho a políticos específicos como trump, clinton, obama. Las verdaderas usaban más fuentes institucionales como reuters, government, washington. El TF-IDF encontró 5,000 palabras importantes. Básicamente las falsas son más sensacionalistas con nombres propios y las reales más formales.")
print()
print("¿Qué métricas fueron mejores y por qué?")
print("Este modelo anduvo mucho mejor, 94.50% de precisión. Las métricas estaban súper balanceadas, tanto precision como recall alrededor del 94-95% para ambas clases. Funcionó bien porque tenía muchísimos más datos (44,898 textos vs 1,309 del Titanic) y además Naive Bayes es perfecto para clasificar textos.")
print()
print("¿Cómo mejorarías este modelo para un caso real?")
print("Usaría más palabras en el TF-IDF, tal vez 10,000. También probaría modelos más modernos como BERT que entienden mejor el contexto. Combinaría el título con el texto completo porque a veces el título ya te dice si es falso. Un ensemble de varios algoritmos diferentes también ayudaría.")
print()
print("EJERCICIO 6: ANÁLISIS EXPLORATORIO - NETFLIX")
print()
print("¿Qué variables influyeron más en el catálogo?")
print("Estados Unidos claramente domina con 3,690 producciones, seguido por India con 1,046. El género más común es International Movies con 2,752 títulos. 2019 fue cuando más contenido agregaron. Las películas duran en promedio 100 minutos que es bastante estándar. Rajiv Chilaka es el director más prolífico con 22 obras.")
print()
print("¿Qué insights fueron mejores y por qué?")
print("Se nota la estrategia de Netflix de volverse global, por eso India está segundo y los International Movies dominan. El pico en 2019 muestra cuándo invirtieron más fuerte, después se estabilizó. La duración de 100 minutos indica que siguen lo que funciona en la industria.")
print()
print("¿Cómo mejorarías este análisis para un caso real?")
print("Le falta la parte predictiva. Podría predecir qué rating va a tener una película según el género y país. Hacer clustering para agrupar contenido similar. Analizar los textos de las descripciones para extraer sentimientos. Crear un sistema de recomendación usando las conexiones entre directores y actores.")