import pandas as pd
import numpy as np
import os

# Ruta del archivo (asegúrate de que la ruta sea correcta)
ruta_archivo = r"C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\Detection\sentences_transcriptions.xlsx"

# Ruta para guardar la muestra
ruta_salida = r"C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\Detection\data_bert\muestra_para_etiquetado.xlsx"

# Tamaño de la muestra (ajusta según tus necesidades)
# Para fine-tuning, generalmente se recomienda entre 200-300 filas para un etiquetado inicial
TAMAÑO_MUESTRA = 250

def generar_muestra_aleatoria():
    """
    Genera una muestra aleatoria del archivo Excel y guarda un nuevo Excel 
    con columnas adicionales para etiquetado manual.
    """
    print(f"Cargando archivo desde: {ruta_archivo}")
    
    # Verificar si el archivo existe
    if not os.path.exists(ruta_archivo):
        print(f"Error: El archivo no existe en la ruta especificada: {ruta_archivo}")
        return
    
    try:
        # Cargar el archivo Excel
        df = pd.read_excel(ruta_archivo)
        
        # Información sobre el dataset original
        filas_totales = len(df)
        print(f"Archivo cargado correctamente. Total de filas: {filas_totales}")
        print(f"Columnas disponibles: {', '.join(df.columns)}")
        
        # Seleccionar muestra aleatoria
        if TAMAÑO_MUESTRA >= filas_totales:
            print(f"Advertencia: El tamaño de muestra solicitado ({TAMAÑO_MUESTRA}) es mayor o igual al total de filas ({filas_totales}).")
            print("Se usará todo el dataset.")
            muestra = df.copy()
        else:
            print(f"Seleccionando muestra aleatoria de {TAMAÑO_MUESTRA} filas...")
            muestra = df.sample(n=50)
        
        # Añadir columnas para etiquetado manual
        muestra['products_detected'] = ""
        muestra['attributes_detected'] = ""
        
        # Guardar la muestra en un nuevo archivo Excel
        muestra.to_excel(ruta_salida, index=False)
        print(f"Muestra aleatoria guardada correctamente en: {ruta_salida}")
        print(f"Tamaño de la muestra: {len(muestra)} filas")
        
        # Mostrar algunas filas de ejemplo
        print("\nPrimeras 5 filas de la muestra (para verificación):")
        print(muestra.head(5)[['transcription', 'products_detected', 'attributes_detected']])
        
        # Generar guía de etiquetado
        generar_guia_etiquetado()
        
        return muestra
    
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        return None

def generar_guia_etiquetado():
    """Genera una guía para el proceso de etiquetado."""
    ruta_guia = r"C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\Detection\guia_etiquetado.txt"
    
    guia = """
GUÍA DE ETIQUETADO PARA PRODUCTOS COSMÉTICOS Y ATRIBUTOS

Para cada transcripción:

1. PRODUCTOS COSMÉTICOS:
   - Identifique todos los productos cosméticos mencionados en la transcripción
   - Formato: "nombre_producto (tipo_producto)" - Ej. "Genifit (skincare)"
   - Separe múltiples productos con comas
   - Productos comunes: concealer, contour, bronzer, blush, foundation, mascara, eyeshadow, highlighter, lipstick, etc.
   - Incluya también nombres de marcas cuando se mencionen específicamente

2. ATRIBUTOS:
   - Identifique todos los atributos o cualidades mencionadas sobre los productos
   - Formato: atributos separados por comas - Ej. "glowing, long-lasting"
   - Atributos comunes: tone, shine, shimmer, bright, light, perfect, beautiful, matte, glowy, creamy, pigmented, etc.
   - Incluya adjetivos descriptivos y beneficios mencionados

EJEMPLOS:

| Transcripción | Productos Detectados | Atributos Detectados |
|--------------|---------------------|---------------------|
| "I use the Genifit from Lancôme" | "Genifit (skincare), Lancôme (brand)" | "glowing" |
| "It costs 38 euros, but this concealer is perfect" | "concealer (makeup)" | "perfect" |
| "I love this foundation because it doesn't crease" | "foundation (makeup)" | "no creasing" |
| "Then I wash my face with this cleanser" | "cleanser (skincare)" | "" |

NOTAS IMPORTANTES:
- Si una transcripción no menciona ningún producto cosmético, deje la celda de productos en blanco
- Si no se mencionan atributos, deje la celda de atributos en blanco
- Tenga en cuenta el contexto completo para identificar productos y atributos implícitos
- Las expresiones pueden variar, use su criterio para identificar referencias a productos cosméticos
"""
    
    # Guardar guía como archivo de texto
    try:
        with open(ruta_guia, "w", encoding="utf-8") as f:
            f.write(guia)
        print(f"\nGuía de etiquetado guardada como: {ruta_guia}")
    except Exception as e:
        print(f"Error al guardar la guía: {e}")

if __name__ == "__main__":
    print("Iniciando generación de muestra aleatoria para etiquetado...")
    generar_muestra_aleatoria()
    print("\nProceso completado. Ahora puede abrir el archivo Excel generado y realizar el etiquetado manual.")