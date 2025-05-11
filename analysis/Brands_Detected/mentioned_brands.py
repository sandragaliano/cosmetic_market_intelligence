"""
This script analyzes TikTok transcriptions to detect beauty brand mentions and perform sentiment analysis.
It processes transcriptions from Excel files, identifies beauty brands using regex patterns, and analyzes
the sentiment context around each brand mention.

1. Generates regex patterns for beauty brand detection with aliases
2. Preprocesses transcriptions by splitting them into individual sentences
3. Detects brand mentions in transcriptions
4. Performs sentiment analysis on the context surrounding brand mentions
5. Generates visualization of top brands and their sentiment scores
6. Exports results to Excel files and PNG charts

"""

# DEPENDENCIES
import json
import pandas as pd
import re
import os
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import traceback
import sys

# Add path to utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils

def generate_brands_regex_json():
    """
    Generates a JSON file with regex patterns for beauty brands detection
    
    Returns:
        str: Path to the generated JSON file or None if error
    """
    output_dir = utils.DETECTION_DIR
    output_file = utils.BRANDS_REGEX_JSON
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating brands regex JSON file: {output_file}")
    
    # Dictionary of brand aliases for common beauty brands
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
    
    brands_data = []
    
    for brand, aliases in brand_aliases.items():
        # Create regex pattern including main brand name and all aliases
        aliases_pattern = "|".join([re.escape(brand)] + [re.escape(alias) for alias in aliases])
        regex_pattern = r'\b(' + aliases_pattern + r')\b'
        
        brands_data.append({
            "brand": brand,
            "aliases": aliases,
            "regex": regex_pattern
        })
    
    # Add additional common brands (without specific aliases)
    additional_brands = [
        "L'Oréal", "Maybelline", "Revlon", "CoverGirl", "NYX", "e.l.f.", "Morphe",
        "Glossier", "Laura Mercier", "Clinique", "Shiseido", "Milk Makeup",
        "NUXE", "La Roche-Posay", "CeraVe", "The Ordinary", "Neutrogena", "Vichy",
        "Bioderma", "Avene", "La Mer", "Tatcha", "Glow Recipe", "Summer Fridays",
        "COSRX", "Paula's Choice", "First Aid Beauty", "Origins", "Ole Henriksen",
        "Kiehl's", "Laneige", "Sisley", "Aveda", "Bumble and Bumble"
    ]
    
    for brand in additional_brands:
        # Only add if not already in the list
        if not any(item["brand"] == brand for item in brands_data):
            brands_data.append({
                "brand": brand,
                "aliases": [],
                "regex": r'\b' + re.escape(brand) + r'\b'
            })
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(brands_data, f, indent=2, ensure_ascii=False)
        print(f"JSON file successfully generated with {len(brands_data)} brands.")
        return output_file
    except Exception as e:
        print(f"Error generating JSON file: {e}")
        print(traceback.format_exc())
        # If an older version exists, try to use it
        if os.path.exists(output_file):
            print(f"Using existing file version: {output_file}")
            return output_file
        else:
            print("Could not generate or find brands JSON file.")
            return None

def preprocess_transcriptions():
    """
    Processes the url_data.xlsx file to split transcriptions into sentences
    and generates the sentences_transcriptions.xlsx file
    
    Returns:
        str: Path to the generated sentences file
    """
    input_file = utils.URL_DATA_CLEANED
    output_dir = utils.DETECTION_DIR
    output_file = utils.SENTENCES_TRANSCRIPTIONS_DETECTION
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading original data file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        
        columns = df.columns.tolist()
        print(f"Available columns: {columns}")
        
        url_column = columns[0]
        transcription_column = columns[1]
        
        print(f"Using column '{url_column}' as identifier and '{transcription_column}' as transcription")
        
        new_rows = []
        
        print("Splitting transcriptions into sentences...")
        total_rows = len(df)
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"Processing row {idx + 1} of {total_rows}...")
                
            url = row[url_column]
            transcription = str(row[transcription_column])
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK resources for sentence tokenization...")
                nltk.download('punkt', quiet=True)
            sentences = sent_tokenize(transcription)
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                if sentence:
                    new_rows.append({
                        url_column: url,
                        transcription_column: sentence
                    })
        
        sentences_df = pd.DataFrame(new_rows)
        
        print(f"Saving processed file: {output_file}")
        sentences_df.to_excel(output_file, index=False)
        
        print(f"Processing completed. Generated {len(new_rows)} sentences from {total_rows} transcriptions.")
        return output_file
        
    except Exception as e:
        print(f"Error processing file: {e}")
        print(traceback.format_exc())
        # Return expected file path anyway
        return output_file

def main():
    """
    Main function that orchestrates the brand detection and sentiment analysis process
    """
    brands_file = generate_brands_regex_json()
    
    # Generate sentences transcriptions file
    transcriptions_file = preprocess_transcriptions()
    
    # Verify necessary files were generated correctly
    if not brands_file:
        print("Error: Could not generate or find brands file.")
        return
    
    if not os.path.exists(transcriptions_file):
        print(f"Error: Could not find transcriptions file: {transcriptions_file}")
        return
    
    # Configure NLTK for sentiment analysis
    try:
        nltk.download('vader_lexicon', quiet=True)
        sentiment_analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Error configuring sentiment analyzer: {e}")
        print("Sentiment analysis will not be available.")
        sentiment_analyzer = None

    # File paths
    output_file = utils.RESULTS_BRAND_DETECTION
    
    # Dictionary of brand aliases - needed for detection
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
    
    # Invert alias dictionary for quick lookup
    alias_to_brand = {}
    for brand, aliases in brand_aliases.items():
        for alias in aliases:
            alias_to_brand[alias.lower()] = brand

    # Load JSON with brands and products
    print(f"Loading brands from {brands_file}...")
    try:
        with open(brands_file, 'r', encoding='utf-8') as f:
            brands_data = json.load(f)
        print(f"Successfully loaded {len(brands_data)} brands from JSON file.")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        print("Continuing with brands defined in script only...")
        brands_data = []
    
    # Create dictionary of regex patterns per brand
    brand_patterns = {}
    
    # Add brands from JSON (priority over script-defined)
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
    
    # Add patterns for any brand not found in JSON
    for brand, aliases in brand_aliases.items():
        if brand not in brand_patterns:
            # Pattern for main brand name
            brand_patterns[brand] = re.compile(r'\b' + re.escape(brand) + r'\b', re.IGNORECASE)
            
            # Patterns for aliases (only if no pattern exists for this brand)
            for alias in aliases:
                # For initials or abbreviations, ensure it's a complete word
                if len(alias) <= 3:
                    pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                else:
                    pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                
                # Save pattern with main brand name
                if brand not in brand_patterns:
                    brand_patterns[brand] = pattern
    
    print(f"Prepared {len(brand_patterns)} regex patterns for brand detection.")
    
    # Load Excel with transcriptions
    print(f"Loading transcriptions from {transcriptions_file}...")
    df = pd.read_excel(transcriptions_file)
    
    # Identify columns
    columns = df.columns.tolist()
    print(f"Detected columns: {columns}")
    
    # Assume first column is URL and second is transcription
    url_column = columns[0]
    transcription_column = columns[1]
    
    # Group transcriptions by URL for contextual search
    print("Grouping transcriptions by URL...")
    urls_to_indices = defaultdict(list)
    urls_to_transcriptions = defaultdict(list)
    
    # Build dictionaries to group by URL
    for index, row in df.iterrows():
        url = row[url_column]
        transcription = str(row[transcription_column])
        
        urls_to_indices[url].append(index)
        urls_to_transcriptions[url].append((index, transcription))
    
    # Create new DataFrame for results including all original rows
    results_df = df.copy()
    
    # Add columns for brand and sentiment information
    results_df['brand_detected'] = 'No brand detected'
    results_df['match_text'] = 'N/A'
    results_df['sentiment_score'] = 0.0     # Sentiment score
    results_df['sentiment_label'] = 'neutral' # Sentiment label
    results_df['context_text'] = 'N/A'      # Context text (2 sentences before and after)
    
    # Variable to count detected brands
    count_detected = 0
    
    # Process each transcription
    print("Analyzing transcriptions for brand detection and sentiment...")
    total_rows = len(df)
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"Processing row {index + 1} of {total_rows}...")
            
        url = row[url_column]
        transcription = str(row[transcription_column])
        transcription_lower = transcription.lower()
        detected = False
        
        # Search for each brand in transcription
        for brand, pattern in brand_patterns.items():
            # Search for pattern matches
            match = pattern.search(transcription)
            
            # Also search for alias matches not in pattern
            alias_found = False
            if not match and brand in brand_aliases:
                for alias in brand_aliases[brand]:
                    if re.search(r'\b' + re.escape(alias) + r'\b', transcription, re.IGNORECASE):
                        match = re.search(r'\b' + re.escape(alias) + r'\b', transcription, re.IGNORECASE)
                        alias_found = True
                        break
            
            if match:
                # Found a brand, update corresponding row
                results_df.at[index, 'brand_detected'] = brand
                
                # Capture context (15 characters before and after match)
                start = max(0, match.start() - 15)
                end = min(len(transcription), match.end() + 15)
                match_context = transcription[start:end]
                results_df.at[index, 'match_text'] = match_context
                
                # Sentiment analysis with context (up to 2 sentences before and after)
                if sentiment_analyzer:
                    try:
                        # Get all transcriptions for this URL with indices
                        url_trans_with_indices = urls_to_transcriptions[url]
                        
                        # Find current transcription index in list
                        current_idx = -1
                        for i, (idx, _) in enumerate(url_trans_with_indices):
                            if idx == index:
                                current_idx = i
                                break
                        
                        if current_idx != -1:
                            # Define range for context search (2 sentences before and after)
                            start_idx = max(0, current_idx - 2)
                            end_idx = min(len(url_trans_with_indices), current_idx + 3)  # +3 because range is exclusive at end
                            
                            # Get context transcriptions
                            context_transcriptions = [url_trans_with_indices[i][1] for i in range(start_idx, end_idx)]
                            context_text = " ".join(context_transcriptions)
                            
                            # Save context text
                            results_df.at[index, 'context_text'] = context_text
                            
                            # Analyze context sentiment
                            sentiment_scores = sentiment_analyzer.polarity_scores(context_text)
                            compound_score = sentiment_scores['compound']
                            
                            # Assign score and label
                            results_df.at[index, 'sentiment_score'] = compound_score
                            
                            # Assign sentiment label
                            if compound_score >= 0.05:
                                sentiment_label = 'positive'
                            elif compound_score <= -0.05:
                                sentiment_label = 'negative'
                            else:
                                sentiment_label = 'neutral'
                            
                            results_df.at[index, 'sentiment_label'] = sentiment_label
                    except Exception as e:
                        print(f"Error in sentiment analysis for index {index}: {e}")
                        print(traceback.format_exc())
                
                detected = True
                count_detected += 1
                break  # Only save first brand detected per transcription
        
        # Search for alias matches directly in text 
        if not detected:
            for alias, brand in alias_to_brand.items():
                if re.search(r'\b' + re.escape(alias) + r'\b', transcription_lower):
                    match = re.search(r'\b' + re.escape(alias) + r'\b', transcription_lower)
                    
                    results_df.at[index, 'brand_detected'] = brand
                    
                    # Capture context
                    start = max(0, match.start() - 15)
                    end = min(len(transcription_lower), match.end() + 15)
                    match_context = transcription_lower[start:end]
                    results_df.at[index, 'match_text'] = match_context
                    
                    # Sentiment analysis with context (up to 2 sentences before and after)
                    if sentiment_analyzer:
                        try:
                            # Get all transcriptions for this URL with indices
                            url_trans_with_indices = urls_to_transcriptions[url]
                            
                            # Find current transcription index in list
                            current_idx = -1
                            for i, (idx, _) in enumerate(url_trans_with_indices):
                                if idx == index:
                                    current_idx = i
                                    break
                            
                            if current_idx != -1:
                                # Define range for context search (2 sentences before and after)
                                start_idx = max(0, current_idx - 2)
                                end_idx = min(len(url_trans_with_indices), current_idx + 3)  # +3 because range is exclusive at end
                                
                                # Get context transcriptions
                                context_transcriptions = [url_trans_with_indices[i][1] for i in range(start_idx, end_idx)]
                                context_text = " ".join(context_transcriptions)
                                
                                # Save context text
                                results_df.at[index, 'context_text'] = context_text
                                
                                # Analyze context sentiment
                                sentiment_scores = sentiment_analyzer.polarity_scores(context_text)
                                compound_score = sentiment_scores['compound']
                                
                                # Assign score and label
                                results_df.at[index, 'sentiment_score'] = compound_score
                                
                                # Assign sentiment label
                                if compound_score >= 0.05:
                                    sentiment_label = 'positive'
                                elif compound_score <= -0.05:
                                    sentiment_label = 'negative'
                                else:
                                    sentiment_label = 'neutral'
                                
                                results_df.at[index, 'sentiment_label'] = sentiment_label
                        except Exception as e:
                            print(f"Error in sentiment analysis for index {index}: {e}")
                            print(traceback.format_exc())
                    
                    count_detected += 1
                    detected = True
                    break
    
    # Report results
    print(f"Found brands in {count_detected} of {len(df)} transcriptions.")
    
    # Save results to Excel
    print(f"Saving results to {output_file}...")
    results_df.to_excel(output_file, index=False)
    
    # Results analysis: Top 10 brands by mentions and their average sentiment
    print("Generating chart of most mentioned brands and their sentiment...")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Filter only rows where brand was detected
        detected_brands_df = results_df[results_df['brand_detected'] != 'No brand detected']
        
        # Count mentions per brand
        brand_counts = detected_brands_df['brand_detected'].value_counts()
        
        # Calculate average sentiment per brand
        brand_sentiment = detected_brands_df.groupby('brand_detected')['sentiment_score'].mean()
        
        # Select top 10 brands by mentions
        top_10_brands = brand_counts.head(10)
        
        # Get sentiment for those same brands
        top_10_sentiment = brand_sentiment[top_10_brands.index]
        
        # Create dataframe to verify results
        results_summary = pd.DataFrame({
            'brand': top_10_brands.index,
            'mentions': top_10_brands.values,
            'avg_sentiment': top_10_sentiment.values
        })
        
        # Save summary for verification
        summary_file = utils.BRAND_MENTIONS_SUMMARY
        results_summary.to_excel(summary_file, index=False)
        print(f"Mentions and sentiment summary saved to: {summary_file}")
        
        # Configure plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Bar positions
        x = np.arange(len(top_10_brands))
        width = 0.35
        
        # Create mention bars
        bars1 = ax.bar(x - width/2, top_10_brands, width, label='Number of mentions', color='skyblue')
        
        # Scale sentiment values for better visualization
        scale_factor = top_10_brands.max() / (top_10_sentiment.max() - top_10_sentiment.min()) * 0.8
        sentiment_scaled = (top_10_sentiment * scale_factor)
        
        # Create sentiment bars
        bars2 = ax.bar(x + width/2, sentiment_scaled, width, 
                    label=f'Average sentiment (scaled {scale_factor:.1f}x)', 
                    color='lightgreen')
        
        # Add labels, title and legend
        ax.set_xlabel('Brands', fontsize=12)
        ax.set_ylabel('Number of mentions', fontsize=12)
        ax.set_title('Top 10 Most Mentioned Brands and Their Average Sentiment', fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(top_10_brands.index, rotation=45, ha='right', fontsize=10)
        ax.legend()
        
        # Add secondary Y scale for sentiment
        ax2 = ax.twinx()
        ax2.set_ylabel('Average sentiment', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')
        # Configure secondary Y axis limits
        max_scaled = sentiment_scaled.max()
        min_scaled = sentiment_scaled.min()
        margin = (max_scaled - min_scaled) * 0.1
        ax2.set_ylim([min_scaled/scale_factor - margin/scale_factor, 
                    max_scaled/scale_factor + margin/scale_factor])
        
        # Add values on mention bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        
        # Add real sentiment values (not scaled) on bars
        for bar, sentiment in zip(bars2, top_10_sentiment):
            height = bar.get_height()
            ax2.annotate(f'{sentiment:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Adjust layout and save
        plt.tight_layout()
        graph_output_file = utils.TOP_BRANDS_GRAPH
        plt.savefig(graph_output_file)
        plt.close()
        
        print(f"Chart saved to: {graph_output_file}")
    except Exception as e:
        print(f"Error generating chart: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()