import json
import pandas as pd
import re
import os
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import traceback

def generate_brands_regex_json():
    """
    Genera el archivo JSON con patrones regex para marcas
    """
    output_dir = r'C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\Detection'
    output_file = os.path.join(output_dir, 'all_brands_products_regex.json')
    
    # Asegurar que el directorio existe
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generando archivo JSON de marcas con regex: {output_file}")
    
    # Diccionario de aliases para marcas comunes
    brand_aliases = {
        "Rare Beauty by Selena Gomez": ["Rare Beauty","Rare"],
        "Charlotte Tilbury": ["Charlotte", "Tilbury"],
        "Patrick Ta Beauty": ["Patrick Ta"],
        "Pat McGrath Labs": ["Pat McGrath", "Pat", "PMG"],
        "Tom Ford Beauty": ["Tom Ford"],
        "Anastasia Beverly Hills": ["ABH", "Anastasia"],
        "Kosas": ["Kosas Cosmetics"],
        "Bobbi Brown": ["Bobbi"],
        "Natasha Denona": ["Natasha", "ND"],
        "Sol de Janeiro": ["Sol", "SDJ"],
        "Drunk Elephant": ["Drunk E", "DE"],
        "Kiehl's": ["Kiehls"],
        "Hourglass": ["Hourglass Cosmetics"],
        "Olaplex": ["Olaplex Hair"],
        "Ouai": ["Ouai Hair"],
        "Make Up By Mario": ["Mario", "MUBM"],
        "Make Up For Ever": ["MUFE"],
        "Huda Beauty": ["Huda"],
        "Fenty Beauty": ["Fenty", "Rihanna Beauty"],
        "MAC Cosmetics": ["MAC"],
        "Urban Decay": ["UD"],
        "Too Faced": ["TF"],
        "Dior Beauty": ["Dior"],
        "Chanel Beauty": ["Chanel"],
        "Gucci Beauty": ["Gucci"],
        "Lancôme": ["Lancome"],
        "Estée Lauder": ["Estee Lauder"],
        "YSL Beauty": ["YSL", "Yves Saint Laurent"],
        "Benefit Cosmetics": ["Benefit"],
        "Giorgio Armani Beauty": ["Armani Beauty"],
        "NARS": ["NARS Cosmetics"],
        "Tarte": ["Tarte Cosmetics"]
    }
    
    # Preparar el JSON con información de regex
    brands_data = []
    
    for brand, aliases in brand_aliases.items():
        # Crear un patrón regex que incluya la marca principal y todos sus aliases
        aliases_pattern = "|".join([re.escape(brand)] + [re.escape(alias) for alias in aliases])
        regex_pattern = r'\b(' + aliases_pattern + r')\b'
        
        # Añadir la entrada al JSON
        brands_data.append({
            "brand": brand,
            "aliases": aliases,
            "regex": regex_pattern
        })
    
    # Añadir algunas marcas adicionales comunes (sin aliases específicos)
    additional_brands = [
        "L'Oréal", "Maybelline", "Revlon", "CoverGirl", "NYX", "e.l.f.", "Morphe",
        "Glossier", "Laura Mercier", "Clinique", "Shiseido", "Milk Makeup",
        "NUXE", "La Roche-Posay", "CeraVe", "The Ordinary", "Neutrogena", "Vichy",
        "Bioderma", "Avene", "La Mer", "Tatcha", "Glow Recipe", "Summer Fridays",
        "COSRX", "Paula's Choice", "First Aid Beauty", "Origins", "Ole Henriksen",
        "Kiehl's", "Laneige", "Sisley", "Aveda", "Bumble and Bumble"
    ]
    
    for brand in additional_brands:
        # Solo añadir si no está ya en la lista
        if not any(item["brand"] == brand for item in brands_data):
            brands_data.append({
                "brand": brand,
                "aliases": [],
                "regex": r'\b' + re.escape(brand) + r'\b'
            })
    
    # Guardar el JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(brands_data, f, indent=2, ensure_ascii=False)
        print(f"Archivo JSON generado exitosamente con {len(brands_data)} marcas.")
        return output_file
    except Exception as e:
        print(f"Error al generar el archivo JSON: {e}")
        print(traceback.format_exc())
        # Si existe una versión anterior del archivo, intentamos usarla
        if os.path.exists(output_file):
            print(f"Se utilizará la versión existente del archivo: {output_file}")
            return output_file
        else:
            print("No se pudo generar ni encontrar el archivo JSON de marcas.")
            return None

def preprocess_transcriptions():
    """
    Procesa el archivo url_data.xlsx para separar las transcripciones en frases
    y generar el archivo sentences_transcriptions.xlsx
    """
    # Rutas de archivos
    input_file = r'C:\Users\sandr\Documents\scrp_tiktok_tfg\data\clean_data\url_data_cleaned.xlsx'
    output_dir = r'C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\Detection'
    output_file = os.path.join(output_dir, 'sentences_transcriptions.xlsx')
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Leyendo el archivo de datos original: {input_file}")
    try:
        # Leer el archivo Excel
        df = pd.read_excel(input_file)
        
        # Verificar las columnas disponibles
        columns = df.columns.tolist()
        print(f"Columnas disponibles: {columns}")
        
        # Asumiendo que la primera columna es la URL y la segunda es la transcripción
        url_column = columns[0]
        transcription_column = columns[1]
        
        print(f"Usando columna '{url_column}' como identificador y '{transcription_column}' como transcripción")
        
        # Crear una lista para almacenar las nuevas filas
        new_rows = []
        
        # Para cada fila en el DataFrame original
        print("Separando transcripciones en frases...")
        total_rows = len(df)
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"Procesando fila {idx + 1} de {total_rows}...")
                
            url = row[url_column]
            transcription = str(row[transcription_column])
            
            # Descargar el tokenizador de oraciones si es necesario
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Descargando recursos de NLTK para tokenización de oraciones...")
                nltk.download('punkt', quiet=True)
            
            # Separar la transcripción en oraciones
            sentences = sent_tokenize(transcription)
            
            # Agregar cada oración como una nueva fila
            for sentence in sentences:
                # Eliminar espacios en blanco innecesarios
                sentence = sentence.strip()
                
                # Solo agregar si la oración no está vacía
                if sentence:
                    new_rows.append({
                        url_column: url,
                        transcription_column: sentence
                    })
        
        # Crear un nuevo DataFrame con las oraciones separadas
        sentences_df = pd.DataFrame(new_rows)
        
        # Guardar el nuevo DataFrame en un archivo Excel
        print(f"Guardando archivo procesado: {output_file}")
        sentences_df.to_excel(output_file, index=False)
        
        print(f"Procesamiento completado. Se generaron {len(new_rows)} frases a partir de {total_rows} transcripciones.")
        return output_file
        
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        print(traceback.format_exc())
        # Devolver la ruta del archivo que esperamos que exista de todos modos
        return r'C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\Detection\sentences_transcriptions.xlsx'

def main():
    # Generar el archivo JSON con patrones regex para marcas
    brands_file = generate_brands_regex_json()
    
    # Generar el archivo de transcripciones por frases
    transcriptions_file = preprocess_transcriptions()
    
    # Verificar que los archivos necesarios se generaron correctamente
    if not brands_file:
        print("Error: No se pudo generar o encontrar el archivo de marcas.")
        return
    
    if not os.path.exists(transcriptions_file):
        print(f"Error: No se pudo encontrar el archivo de transcripciones: {transcriptions_file}")
        return
    
    # Configuración de NLTK para análisis de sentimiento
    try:
        nltk.download('vader_lexicon', quiet=True)
        sentiment_analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Error al configurar el analizador de sentimiento: {e}")
        print("El análisis de sentimiento no estará disponible.")
        sentiment_analyzer = None

    # Rutas de archivos
    output_file = r'C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\Detection\results_brand_detection.xlsx'
    
    # Diccionario de aliases para marcas comunes - necesario para la detección
    brand_aliases = {
        "Rare Beauty by Selena Gomez": ["Rare Beauty","Rare"],
        "Charlotte Tilbury": ["Charlotte", "Tilbury"],
        "Patrick Ta Beauty": ["Patrick Ta"],
        "Pat McGrath Labs": ["Pat McGrath", "Pat", "PMG"],
        "Tom Ford Beauty": ["Tom Ford"],
        "Anastasia Beverly Hills": ["ABH", "Anastasia"],
        "Kosas": ["Kosas Cosmetics"],
        "Bobbi Brown": ["Bobbi"],
        "Natasha Denona": ["Natasha", "ND"],
        "Sol de Janeiro": ["Sol", "SDJ"],
        "Drunk Elephant": ["Drunk E", "DE"],
        "Kiehl's": ["Kiehls"],
        "Hourglass": ["Hourglass Cosmetics"],
        "Olaplex": ["Olaplex Hair"],
        "Ouai": ["Ouai Hair"],
        "Make Up By Mario": ["Mario", "MUBM"],
        "Make Up For Ever": ["MUFE"],
        "Huda Beauty": ["Huda"],
        "Fenty Beauty": ["Fenty", "Rihanna Beauty"],
        "MAC Cosmetics": ["MAC"],
        "Urban Decay": ["UD"],
        "Too Faced": ["TF"],
        "Dior Beauty": ["Dior"],
        "Chanel Beauty": ["Chanel"],
        "Gucci Beauty": ["Gucci"],
        "Lancôme": ["Lancome"],
        "Estée Lauder": ["Estee Lauder"],
        "YSL Beauty": ["YSL", "Yves Saint Laurent"],
        "Benefit Cosmetics": ["Benefit"],
        "Giorgio Armani Beauty": ["Armani Beauty"],
        "NARS": ["NARS Cosmetics"],
        "Tarte": ["Tarte Cosmetics"],
        "SEPHORA COLLECTION": ["Sephora Collection"]
    }
    
    # Invertir el diccionario de aliases para búsqueda rápida
    alias_to_brand = {}
    for brand, aliases in brand_aliases.items():
        for alias in aliases:
            alias_to_brand[alias.lower()] = brand

    # Cargar el JSON con marcas y productos
    print(f"Cargando marcas desde {brands_file}...")
    try:
        with open(brands_file, 'r', encoding='utf-8') as f:
            brands_data = json.load(f)
        print(f"Se cargaron correctamente {len(brands_data)} marcas del archivo JSON.")
    except Exception as e:
        print(f"Error al cargar el archivo JSON: {e}")
        print("Continuando solo con las marcas definidas en el script...")
        brands_data = []
    
    # Crear un diccionario de expresiones regulares por marca
    brand_patterns = {}
    
    # Añadir marcas del JSON (prioridad sobre las definidas en el script)
    if isinstance(brands_data, list):
        for item in brands_data:
            if isinstance(item, dict) and 'brand' in item:
                brand = item['brand']
                if 'regex' in item:
                    regex = item['regex']
                    brand_patterns[brand] = re.compile(regex, re.IGNORECASE)
                else:
                    regex = r'\b' + re.escape(brand) + r'\b'
                    brand_patterns[brand] = re.compile(regex, re.IGNORECASE)
    
    # Añadir patrones para cualquier marca que no se haya encontrado en el JSON
    for brand, aliases in brand_aliases.items():
        if brand not in brand_patterns:
            # Patrón para el nombre principal de la marca
            brand_patterns[brand] = re.compile(r'\b' + re.escape(brand) + r'\b', re.IGNORECASE)
            
            # Patrones para los aliases (solo si no existe ya un patrón para esta marca)
            for alias in aliases:
                # Si el alias es solo una inicial o abreviatura, aseguramos que sea una palabra completa
                if len(alias) <= 3:
                    pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                else:
                    pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                
                # Guardamos el patrón con el nombre de la marca principal
                if brand not in brand_patterns:
                    brand_patterns[brand] = pattern
    
    print(f"Se prepararon {len(brand_patterns)} patrones de regex para la detección de marcas.")
    
    # Cargar el Excel con transcripciones
    print(f"Cargando transcripciones desde {transcriptions_file}...")
    df = pd.read_excel(transcriptions_file)
    
    # Identificar las columnas
    columns = df.columns.tolist()
    print(f"Columnas detectadas: {columns}")
    
    # Asumimos que la primera columna es la URL y la segunda es la transcripción
    url_column = columns[0]
    transcription_column = columns[1]
    
    # Agrupar transcripciones por URL para búsqueda contextual
    print("Agrupando transcripciones por URL...")
    urls_to_indices = defaultdict(list)
    urls_to_transcriptions = defaultdict(list)
    
    # Construir diccionarios para agrupar por URL
    for index, row in df.iterrows():
        url = row[url_column]
        transcription = str(row[transcription_column])
        
        urls_to_indices[url].append(index)
        urls_to_transcriptions[url].append((index, transcription))
    
    # Crear un nuevo DataFrame para los resultados que incluirá todas las filas originales
    results_df = df.copy()
    
    # Agregamos columnas para la información de marcas y sentimiento
    results_df['brand_detected'] = 'No se detectó marca'
    results_df['match_text'] = 'N/A'
    results_df['sentiment_score'] = 0.0     # Puntuación de sentimiento
    results_df['sentiment_label'] = 'neutral' # Etiqueta de sentimiento
    results_df['context_text'] = 'N/A'      # Texto del contexto (2 frases arriba y abajo)
    
    # Variable para contar marcas detectadas
    count_detected = 0
    
    # Procesar cada transcripción
    print("Analizando transcripciones para detectar marcas y sentimiento...")
    total_rows = len(df)
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"Procesando fila {index + 1} de {total_rows}...")
            
        url = row[url_column]
        transcription = str(row[transcription_column])
        transcription_lower = transcription.lower()
        detected = False
        
        # Buscar cada marca en la transcripción
        for brand, pattern in brand_patterns.items():
            # Buscar coincidencias del patrón
            match = pattern.search(transcription)
            
            # También buscar coincidencias de aliases que puedan no estar en el patrón
            alias_found = False
            if not match and brand in brand_aliases:
                for alias in brand_aliases[brand]:
                    if re.search(r'\b' + re.escape(alias) + r'\b', transcription, re.IGNORECASE):
                        match = re.search(r'\b' + re.escape(alias) + r'\b', transcription, re.IGNORECASE)
                        alias_found = True
                        break
            
            if match:
                # Encontramos una marca, actualizamos la fila correspondiente
                results_df.at[index, 'brand_detected'] = brand
                
                # Capturar el contexto (15 caracteres antes y después de la coincidencia)
                start = max(0, match.start() - 15)
                end = min(len(transcription), match.end() + 15)
                match_context = transcription[start:end]
                results_df.at[index, 'match_text'] = match_context
                
                # Análisis de sentimiento con contexto (hasta 2 frases arriba y abajo)
                if sentiment_analyzer:
                    try:
                        # Obtener todas las transcripciones para esta URL con sus índices
                        url_trans_with_indices = urls_to_transcriptions[url]
                        
                        # Encontrar el índice de la transcripción actual en la lista de transcripciones
                        current_idx = -1
                        for i, (idx, _) in enumerate(url_trans_with_indices):
                            if idx == index:
                                current_idx = i
                                break
                        
                        if current_idx != -1:
                            # Definir rango para buscar contexto (2 frases arriba y abajo)
                            start_idx = max(0, current_idx - 2)
                            end_idx = min(len(url_trans_with_indices), current_idx + 3)  # +3 porque el rango es exclusivo al final
                            
                            # Obtener transcripciones del contexto
                            context_transcriptions = [url_trans_with_indices[i][1] for i in range(start_idx, end_idx)]
                            context_text = " ".join(context_transcriptions)
                            
                            # Guardar el texto del contexto
                            results_df.at[index, 'context_text'] = context_text
                            
                            # Analizar sentimiento del contexto
                            sentiment_scores = sentiment_analyzer.polarity_scores(context_text)
                            compound_score = sentiment_scores['compound']
                            
                            # Asignar puntuación y etiqueta
                            results_df.at[index, 'sentiment_score'] = compound_score
                            
                            # Asignar etiqueta de sentimiento
                            if compound_score >= 0.05:
                                sentiment_label = 'positive'
                            elif compound_score <= -0.05:
                                sentiment_label = 'negative'
                            else:
                                sentiment_label = 'neutral'
                            
                            results_df.at[index, 'sentiment_label'] = sentiment_label
                    except Exception as e:
                        print(f"Error en el análisis de sentimiento para el índice {index}: {e}")
                        print(traceback.format_exc())
                
                detected = True
                count_detected += 1
                break  # Solo guardamos la primera marca detectada por transcripción
        
        # Buscar coincidencias de alias directamente en el texto 
        if not detected:
            for alias, brand in alias_to_brand.items():
                if re.search(r'\b' + re.escape(alias) + r'\b', transcription_lower):
                    match = re.search(r'\b' + re.escape(alias) + r'\b', transcription_lower)
                    
                    results_df.at[index, 'brand_detected'] = brand
                    
                    # Capturar el contexto
                    start = max(0, match.start() - 15)
                    end = min(len(transcription_lower), match.end() + 15)
                    match_context = transcription_lower[start:end]
                    results_df.at[index, 'match_text'] = match_context
                    
                    # Análisis de sentimiento con contexto (hasta 2 frases arriba y abajo)
                    if sentiment_analyzer:
                        try:
                            # Obtener todas las transcripciones para esta URL con sus índices
                            url_trans_with_indices = urls_to_transcriptions[url]
                            
                            # Encontrar el índice de la transcripción actual en la lista de transcripciones
                            current_idx = -1
                            for i, (idx, _) in enumerate(url_trans_with_indices):
                                if idx == index:
                                    current_idx = i
                                    break
                            
                            if current_idx != -1:
                                # Definir rango para buscar contexto (2 frases arriba y abajo)
                                start_idx = max(0, current_idx - 2)
                                end_idx = min(len(url_trans_with_indices), current_idx + 3)  # +3 porque el rango es exclusivo al final
                                
                                # Obtener transcripciones del contexto
                                context_transcriptions = [url_trans_with_indices[i][1] for i in range(start_idx, end_idx)]
                                context_text = " ".join(context_transcriptions)
                                
                                # Guardar el texto del contexto
                                results_df.at[index, 'context_text'] = context_text
                                
                                # Analizar sentimiento del contexto
                                sentiment_scores = sentiment_analyzer.polarity_scores(context_text)
                                compound_score = sentiment_scores['compound']
                                
                                # Asignar puntuación y etiqueta
                                results_df.at[index, 'sentiment_score'] = compound_score
                                
                                # Asignar etiqueta de sentimiento
                                if compound_score >= 0.05:
                                    sentiment_label = 'positive'
                                elif compound_score <= -0.05:
                                    sentiment_label = 'negative'
                                else:
                                    sentiment_label = 'neutral'
                                
                                results_df.at[index, 'sentiment_label'] = sentiment_label
                        except Exception as e:
                            print(f"Error en el análisis de sentimiento para el índice {index}: {e}")
                            print(traceback.format_exc())
                    
                    count_detected += 1
                    detected = True
                    break
    
    # Informar resultados
    print(f"Se encontraron marcas en {count_detected} de {len(df)} transcripciones.")
    
    # Guardar resultados en Excel
    print(f"Guardando resultados en {output_file}...")
    results_df.to_excel(output_file, index=False)
    # Análisis de resultados: Top 10 marcas por menciones y su sentimiento promedio
    print("Generando gráfico de las marcas más mencionadas y su sentimiento...")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Filtrar solo las filas donde se detectó una marca
        detected_brands_df = results_df[results_df['brand_detected'] != 'No se detectó marca']
        
        # Contar menciones por marca
        brand_counts = detected_brands_df['brand_detected'].value_counts()
        
        # Calcular sentimiento promedio por marca (asegurarnos que hacemos la agrupación correctamente)
        brand_sentiment = detected_brands_df.groupby('brand_detected')['sentiment_score'].mean()
        
        # Seleccionar el top 10 marcas por menciones
        top_10_brands = brand_counts.head(10)
        
        # Obtener sentimiento para esas mismas marcas
        top_10_sentiment = brand_sentiment[top_10_brands.index]
        
        # Crear un dataframe para verificar los resultados
        results_summary = pd.DataFrame({
            'brand': top_10_brands.index,
            'mentions': top_10_brands.values,
            'avg_sentiment': top_10_sentiment.values
        })
        
        # Guardar este resumen para verificación
        summary_file = os.path.join(os.path.dirname(output_file), 'brand_mentions_sentiment_summary.xlsx')
        results_summary.to_excel(summary_file, index=False)
        print(f"Resumen de menciones y sentimiento guardado en: {summary_file}")
        
        # Configurar el gráfico
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Posiciones de las barras
        x = np.arange(len(top_10_brands))
        width = 0.35
        
        # Crear las barras de menciones
        bars1 = ax.bar(x - width/2, top_10_brands, width, label='Número de menciones', color='skyblue')
        
        # Escalar los valores de sentimiento para mejor visualización
        # Usar un factor que haga visible las barras de sentimiento junto a las de menciones
        scale_factor = top_10_brands.max() / (top_10_sentiment.max() - top_10_sentiment.min()) * 0.8
        sentiment_scaled = (top_10_sentiment * scale_factor)
        
        # Crear las barras de sentimiento
        bars2 = ax.bar(x + width/2, sentiment_scaled, width, 
                    label=f'Sentimiento promedio (escalado {scale_factor:.1f}x)', 
                    color='lightgreen')
        
        # Añadir etiquetas, título y leyenda
        ax.set_xlabel('Marcas', fontsize=12)
        ax.set_ylabel('Número de menciones', fontsize=12)
        ax.set_title('Top 10 Marcas Más Mencionadas y su Sentimiento Promedio', fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(top_10_brands.index, rotation=45, ha='right', fontsize=10)
        ax.legend()
        
        # Añadir una segunda escala Y para el sentimiento
        ax2 = ax.twinx()
        ax2.set_ylabel('Sentimiento promedio', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')
        # Configurar los límites del eje Y secundario
        max_scaled = sentiment_scaled.max()
        min_scaled = sentiment_scaled.min()
        margin = (max_scaled - min_scaled) * 0.1
        ax2.set_ylim([min_scaled/scale_factor - margin/scale_factor, 
                    max_scaled/scale_factor + margin/scale_factor])
        
        # Añadir valores en las barras de menciones
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        
        # Añadir valores de sentimiento real (no escalado) en las barras
        for bar, sentiment in zip(bars2, top_10_sentiment):
            height = bar.get_height()
            ax2.annotate(f'{sentiment:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        graph_output_file = os.path.join(os.path.dirname(output_file), 'top_brands_mentions_sentiment.png')
        plt.savefig(graph_output_file)
        plt.close()
        
        print(f"Gráfico guardado en: {graph_output_file}")
    except Exception as e:
        print(f"Error al generar el gráfico: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()