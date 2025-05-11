# Cosmetic Market Intelligence: Análisis de Tendencias de Consumo de Productos y Características de Éxito en Redes Sociales 

## 📋 Descripción

Este repositorio contiene el código y análisis desarrollados para el Trabajo de Fin de Grado (TFG) titulado **"Cosmetic Market Intelligence: Análisis de Tendencias y Características de Éxito en Redes Sociales"**.

El proyecto analiza el mercado cosmético en plataformas digitales, investigando tendencias y patrones de éxito de productos de Sephora que aparecen en TikTok, proporcionando insights valiosos para el sector de la belleza.

## Sobre uno de los dos dataset principales...

El dataset de Sephora extraído mediante web scraping con [Apify](https://apify.com/) debe descargarse por separado:

📥 **[Descargar Dataset Sephora](https://drive.google.com/file/d/1H1NMBdPpLVVITMEWWj6CJsvhvgXyot96/view?usp=sharing)**

Una vez descargado, colócalo en la carpeta `data/` con el nombre: `dataset_sephora.json`

> ⚠️ **Nota:** Este archivo no está incluido en el repositorio debido a restricciones de tamaño de GitHub.

## Metodología

### 1. Obtención de Datos
- **Web Scraping de TikTok**: Extracción de URLs y metadatos de videos
- **Transcripción Audio-Texto**: Conversión mediante modelo Whisper
- **Scraping de Sephora**: Extracción del catálogo completo vía Apify

### 2. Procesamiento de Datos
- **ETL (Extract, Transform, Load)**: Limpieza y estructuración de datos
- **EDA (Exploratory Data Analysis)**: Análisis exploratorio inicial

### 3. Análisis Avanzado
- **Detección de Marcas**: Identificación mediante expresiones regulares
- **NER Personalizado**: Fine-tuning de DistilBERT para detectar productos y atributos
- **Clustering**: K-Means para segmentación de marcas
- **Market Basket Analysis**: Análisis de patrones de compra conjunta mediante algortimo Apriori

## 📁 Estructura del Repositorio

```
cosmetic-market-intelligence/
│
├── analysis/                 
│   ├── Bert/                     # Detección NER con BERT
│   │   ├── data_bert/            # Datos para el modelo BERT
│   │   ├── detection_results/    # Resultados de detección
│   │   ├── final_model/          # Modelo BERT entrenado
│   │   └── bert_detection.ipynb  # Notebook de detección
│   │
│   ├── Brands_Detected/          # Detección de marcas, sentimiento, etc
│   │   ├── all_brands_products_regex.json
│   │   ├── brand_mentions_sentiment_summary.xlsx
│   │   ├── mentioned_brands.py   # Script de detección
│   │   ├── results_brand_detection.xlsx
│   │   ├── sentences_transcriptions.xlsx
│   │   └── top_brands_mentions_sentiment.png
│   │
│   ├── Detection/                # Scripts de detección adicionales
│   │
│   ├── Kmeans/                   # Clustering de marcas
│   │   ├── data_for_kmeans/      # Datos preparados en fases anteriores para clustering
│   │   ├── Kmeans_results/       # Resultados del clustering
│   │   └── Kmeans.ipynb          # Notebook de K-means
│   │
│   └── MBA/                      # Market Basket Analysis
│       ├── mba_results/          # Resultados del análisis
│       └── mba.ipynb             # Notebook MBA
│
├── data/                         # Datasets
│   ├── clean_data/               # Datos limpios tras proceso ETL
│   │   ├── sephora_website_cleaned.csv
│   │   └── url_data_cleaned.xlsx
│   │
│   ├── etl_and_eda/              # ETL y Análisis Exploratorio
│   │   ├── eda&etl_sephora_store.ipynb
│   │   └── eda&etl_url_data.ipynb
│   │
│   ├── dataset_sephora.json      # Dataset Sephora (descargar externamente y colocar aquí)
│   └── url_data.xlsx             # URLs de TikTok
│
├── utils.py                      # Elementos auxiliares
├── tiktokurl_extraction.py       # Script principal de extracción (scraping de los videos de TikTok)
└── README.md                     # Este archivo
```

## 🛠️ Tecnologías Utilizadas

- **Python**: Lenguaje principal
- **NLP**: SpaCy, NLTK, Transformers (DistilBERT)
- **Machine Learning**: Scikit-learn, MLxtend
- **Web Scraping**: Selenium, BeautifulSoup, Apify
- **Análisis de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn, NetworkX

Entre otras... :)

## 👤 Autor

**Sandra Galiano Bernardino**
- Email: sandra.galiber@gmail.com
- Universidad: Francisco de Vitoria
- Grado: Business Analytics, mención en Ciencia de Datos
- Curso: 2024-2025

## Para Citar Este trabajo

Si utilizas este código o datos en tu investigación, por favor cítame de la siguiente forma:

Galiano Bernardino, S. (2025). Cosmetic Market Intelligence: Análisis de Tendencias y
Características de Éxito en Redes Sociales. Trabajo de Fin de Grado, Universidad Francisco de Vitoria.
