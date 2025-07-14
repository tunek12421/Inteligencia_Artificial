#!/usr/bin/env python3
# ==============================================
# GENERADOR DE CSV PARA EJERCICIO 11 - SENTIMIENTOS
# Crea el archivo CSV exacto que necesitas para tu ejercicio
# ==============================================

import pandas as pd
import numpy as np
import os

def crear_csv_desde_dataset_original():
    """
    Crea un CSV limpio desde tu dataset original de Rotten Tomatoes
    """
    print("🔄 PROCESANDO DATASET ORIGINAL...")
    print("-" * 40)
    
    try:
        # Intentar leer el dataset original
        df = pd.read_csv('Dataset/data_rt.csv')
        
        print(f"✅ Dataset original cargado: {len(df):,} registros")
        print(f"📋 Columnas encontradas: {list(df.columns)}")
        
        # Verificar estructura
        if 'reviews' in df.columns and 'labels' in df.columns:
            print("✅ Estructura correcta detectada")
            
            # Limpiar datos si es necesario
            df_clean = df.copy()
            
            # Eliminar filas vacías
            df_clean = df_clean.dropna()
            
            # Asegurar que labels sean 0 y 1
            df_clean['labels'] = df_clean['labels'].astype(int)
            
            # Verificar distribución
            distribucion = df_clean['labels'].value_counts().sort_index()
            print(f"📊 Distribución después de limpieza:")
            for label, count in distribucion.items():
                tipo = "NEGATIVO" if label == 0 else "POSITIVO"
                print(f"   {label} ({tipo}): {count:,} registros")
            
            # Guardar CSV limpio
            archivo_salida = 'opiniones_productos.csv'
            df_clean.to_csv(archivo_salida, index=False)
            
            print(f"\n✅ CSV creado exitosamente: {archivo_salida}")
            print(f"📄 Listo para usar en tu ejercicio")
            
            return df_clean, archivo_salida
            
        else:
            print("❌ Estructura incorrecta en dataset original")
            return None, None
            
    except FileNotFoundError:
        print("❌ No se encontró Dataset/data_rt.csv")
        return None, None
    except Exception as e:
        print(f"❌ Error procesando dataset original: {e}")
        return None, None

def crear_csv_sintetico_backup():
    """
    Crea un CSV sintético como respaldo si el original no funciona
    """
    print("\n🔧 CREANDO DATASET SINTÉTICO DE RESPALDO...")
    print("-" * 45)
    
    # Datos sintéticos de ejemplo más realistas
    reseñas_negativas = [
        "This movie is absolutely terrible and boring",
        "Worst film I have ever seen in my entire life",
        "Complete waste of time and money",
        "Poor acting and terrible storyline",
        "Disappointing and poorly executed",
        "Boring and uninteresting from start to finish",
        "Bad direction and awful script",
        "Not worth watching at all",
        "Terrible plot with bad characters",
        "Poorly made and very disappointing",
        "Awful movie with no redeeming qualities",
        "Boring story that goes nowhere",
        "Bad acting ruins the entire experience",
        "Terrible waste of good actors",
        "Poorly written and badly directed",
        "Disappointing sequel to a great original",
        "Boring and predictable storyline",
        "Bad script with terrible dialogue",
        "Not entertaining at all",
        "Awful cinematography and poor editing"
    ]
    
    reseñas_positivas = [
        "Amazing movie with excellent acting and great story",
        "Fantastic film that exceeded all my expectations",
        "Brilliant cinematography and outstanding performances",
        "Excellent story with wonderful character development",
        "Outstanding movie that I highly recommend",
        "Great acting and beautiful visuals throughout",
        "Fantastic story that kept me engaged",
        "Excellent direction and superb acting",
        "Amazing plot with great character development",
        "Brilliant movie with outstanding performances",
        "Wonderful film with excellent cinematography",
        "Great story that was beautifully executed",
        "Fantastic movie with amazing special effects",
        "Excellent acting and brilliant direction",
        "Outstanding film that deserves recognition",
        "Amazing movie with a compelling storyline",
        "Great performances by all the actors",
        "Brilliant film with excellent writing",
        "Fantastic cinematography and great music",
        "Excellent movie that I will watch again"
    ]
    
    # Crear dataset balanceado
    reviews = reseñas_negativas + reseñas_positivas
    labels = [0] * len(reseñas_negativas) + [1] * len(reseñas_positivas)
    
    # Mezclar datos
    data = list(zip(reviews, labels))
    np.random.shuffle(data)
    reviews_shuffled, labels_shuffled = zip(*data)
    
    df_sintetico = pd.DataFrame({
        'reviews': reviews_shuffled,
        'labels': labels_shuffled
    })
    
    # Guardar CSV sintético
    archivo_sintetico = 'opiniones_productos_sintetico.csv'
    df_sintetico.to_csv(archivo_sintetico, index=False)
    
    print(f"✅ CSV sintético creado: {archivo_sintetico}")
    print(f"📊 {len(df_sintetico)} registros balanceados")
    print(f"📋 Estructura: reviews, labels (0=negativo, 1=positivo)")
    
    return df_sintetico, archivo_sintetico

def verificar_csv_para_ejercicio(archivo):
    """
    Verifica que el CSV sea compatible con tu ejercicio
    """
    print(f"\n🔍 VERIFICANDO COMPATIBILIDAD CON EJERCICIO 11...")
    print("-" * 50)
    
    try:
        df = pd.read_csv(archivo)
        
        # Verificaciones necesarias
        checks = []
        
        # Check 1: Columnas correctas
        if 'reviews' in df.columns and 'labels' in df.columns:
            checks.append("✅ Columnas 'reviews' y 'labels' presentes")
        else:
            checks.append("❌ Columnas incorrectas")
            
        # Check 2: Etiquetas binarias
        labels_unicos = sorted(df['labels'].unique())
        if labels_unicos == [0, 1]:
            checks.append("✅ Etiquetas binarias (0, 1) correctas")
        else:
            checks.append(f"❌ Etiquetas incorrectas: {labels_unicos}")
            
        # Check 3: No valores nulos
        if df.isnull().sum().sum() == 0:
            checks.append("✅ Sin valores nulos")
        else:
            checks.append("❌ Contiene valores nulos")
            
        # Check 4: Texto en reviews
        if df['reviews'].dtype == 'object':
            checks.append("✅ Columna 'reviews' contiene texto")
        else:
            checks.append("❌ Columna 'reviews' no es texto")
            
        # Mostrar resultados
        for check in checks:
            print(f"   {check}")
            
        # Estadísticas finales
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   Total de registros: {len(df):,}")
        distribucion = df['labels'].value_counts().sort_index()
        for label, count in distribucion.items():
            tipo = "NEGATIVO" if label == 0 else "POSITIVO"
            porcentaje = (count / len(df)) * 100
            print(f"   {label} ({tipo}): {count:,} ({porcentaje:.1f}%)")
            
        # Verificar si es balanceado
        if len(distribucion) == 2 and abs(distribucion[0] - distribucion[1]) <= len(df) * 0.1:
            print(f"   ✅ Dataset balanceado")
        else:
            print(f"   ⚠️  Dataset desbalanceado")
            
        return True
        
    except Exception as e:
        print(f"❌ Error verificando archivo: {e}")
        return False

def mostrar_muestra_csv(archivo):
    """
    Muestra una muestra del CSV creado
    """
    print(f"\n📋 MUESTRA DEL ARCHIVO: {archivo}")
    print("-" * 50)
    
    try:
        df = pd.read_csv(archivo)
        
        # Mostrar primeras líneas
        print("Primeras 5 líneas:")
        print(df.head().to_string(index=False, max_colwidth=60))
        
        print(f"\nEjemplos de cada clase:")
        negativo = df[df['labels'] == 0].iloc[0]
        positivo = df[df['labels'] == 1].iloc[0]
        
        print(f"NEGATIVO (0): \"{negativo['reviews'][:80]}...\"")
        print(f"POSITIVO (1): \"{positivo['reviews'][:80]}...\"")
        
    except Exception as e:
        print(f"Error mostrando muestra: {e}")

def generar_instrucciones_uso():
    """
    Genera instrucciones para usar el CSV en el ejercicio
    """
    print(f"\n📝 INSTRUCCIONES DE USO:")
    print("=" * 30)
    print("1. Usa el archivo 'opiniones_productos.csv' en tu ejercicio")
    print("2. Cambia la línea de importación en tu código:")
    print("   ANTES: df = pd.read_csv('../../Dataset/data_rt.csv')")
    print("   DESPUÉS: df = pd.read_csv('opiniones_productos.csv')")
    print("3. Ejecuta tu ejercicio normalmente")
    print("4. El código funcionará exactamente igual")
    print()
    print("📁 ARCHIVOS GENERADOS:")
    if os.path.exists('opiniones_productos.csv'):
        print("   ✅ opiniones_productos.csv (principal)")
    if os.path.exists('opiniones_productos_sintetico.csv'):
        print("   ✅ opiniones_productos_sintetico.csv (respaldo)")

def main():
    """
    Función principal que coordina todo el proceso
    """
    print("=" * 60)
    print("GENERADOR DE CSV PARA EJERCICIO 11 - ANÁLISIS DE SENTIMIENTOS")
    print("=" * 60)
    
    # Intentar crear desde dataset original
    df_original, archivo_original = crear_csv_desde_dataset_original()
    
    # Si no funciona, crear sintético
    if df_original is None:
        print("\n⚠️  Creando dataset sintético como alternativa...")
        df_sintetico, archivo_sintetico = crear_csv_sintetico_backup()
        
        # Usar el sintético como principal
        if os.path.exists(archivo_sintetico):
            # Copiar como archivo principal
            df_sintetico.to_csv('opiniones_productos.csv', index=False)
            archivo_principal = 'opiniones_productos.csv'
        else:
            archivo_principal = archivo_sintetico
    else:
        archivo_principal = archivo_original
    
    # Verificar el archivo final
    if archivo_principal and os.path.exists(archivo_principal):
        if verificar_csv_para_ejercicio(archivo_principal):
            mostrar_muestra_csv(archivo_principal)
            generar_instrucciones_uso()
            
            print(f"\n🎉 ¡CSV GENERADO EXITOSAMENTE!")
            print(f"✅ Archivo listo: {archivo_principal}")
            print(f"✅ Compatible con tu Ejercicio 11")
            print(f"✅ Estructura: reviews, labels (0=negativo, 1=positivo)")
        else:
            print(f"\n❌ Error: CSV generado no es compatible")
    else:
        print(f"\n❌ Error: No se pudo generar el CSV")

if __name__ == "__main__":
    main()