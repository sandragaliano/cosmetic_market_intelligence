# Cosmetic Market Intelligence
Este repositorio contiene el código y análisis desarrollados para la realización de mi Trabajo de Fin de Grado (TFG) titulado "Cosmetic Market Intelligence: Análisis de Tendencias y Características de Éxito en Redes Sociales". El objetivo de este proyecto es analizar el mercado cosmético en plataformas digitales, centrándose en las tendencias y patrones de éxito de productos ofertados en Sephora que aparecen en TikTok.

# Descripción del Proyecto

Para llevar a cabo el análisis, se han implementado las siguientes técnicas:
1. Scraping de videos de tiktok
1.1. Transcricpiones de audio a texto de los videos con el modelo Whisper. 
3. Scraping de la Web de Sephora a través de Apify
4. ETL y EDA
5. Análisis de Datos
  5.1. Detección de nombres de marcas mencionadas con expresiones regulares
  5.2. Fine-tuning del modelo DistilBert y procesamiento de dataset para encontrar atributos y tipos de productos en las             transcripciones.
   5.3. K-Means para clusterizar las marcas según diferentes características y atributos.
   5.4. Market Basket Analysis (MBA): análisis de productos comprados conjuntamente y reglas de asociación. 

# Contenido del Repositorio

data/ : Contiene los datos scrapeados, en el caso de la web de Sephora desde Apify y en el caso del excel, extraido a través del script tiktokurl_extraction.py. Además contiene una carpeta clean_data con los datos limpios del proceso etl. 

analysis/Detection/: código y datos destinados al análisis y fine-tuning del modelo Distilbert con el fin de crear un NER apropiado para el proyecto. 

analysis/kmeans/: Por una parte contiene una carpeta data for kmeans que contiene los datos preparados para su análisis. Y por otra, tiene una carpeta de resultados donde se van guardo los resultados del proceso K-Means que está recogido en jupyter Kmeans.ipynb

analysis/MBA/: Contiene el código relativo al mba en un archivo jupyter y también una carpeta de resultados. 

Este proyecto ha sido desarrollado por [Sandra Galiano] como parte del TFG en [Universidad Francisco de Vitoria].
