import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import os

# Sentiment analysis thresholds
SENTIMENT_THRESHOLDS = {
    'positive': 0.1,    # Score > 0.1 is positive
    'negative': -0.1,   # Score < -0.1 is negative
    # Between -0.1 and 0.1 is neutral
}

# Technical categories and their related terms
TECH_CATEGORIES = {
    'display_quality': ['OLED', 'QD-OLED', 'HDR', 'SDR', 'color', 'brightness', 'contrast', 
                       'resolution', 'panel', 'calibration', 'RGB'],
    'performance': ['response time', 'refresh rate', 'latency', 'fps', 'gaming', 'lag', 
                   'ghosting', 'tearing'],
    'connectivity': ['USB', 'HDMI', 'DisplayPort', 'DP', 'KVM', 'USB-C', 'port', 'input'],
    'design': ['ergonomic', 'stand', 'build quality', 'design', 'cooling', 'workmanship', 'height', 'width', 'depth', 'weight'],
    'features': ['G-Sync', 'FreeSync', 'HDR', 'PiP', 'PbP', 'OSD'],
    'price': ['budget', 'affordable', 'expensive', 'cost-effective', 'value', 'cost'],
    'sound': ['sound', 'audio', 'speaker', 'subwoofer', 'bass', 'treble', 'volume', 'noise', 'noise cancellation'],
    'camera': ['camera', 'video', 'photo', 'quality', 'resolution', 'zoom', 'focus', 'exposure'],
    'battery': ['battery', 'life', 'charging', 'wireless', 'usb-c', 'usb-pd', 'power delivery'],
}

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import pandas
        import nltk
        import textblob
        import sklearn
        import openpyxl
        return True
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install required packages using:")
        print("pip install -r requirements_advanced.txt")
        return False

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_pros_cons(text, pattern_type='pros'):
    """Extract pros or cons from text based on pattern"""
    if pd.isna(text):
        return []
    
    # Define patterns for pros and cons
    patterns = {
        'pros': [r'\(\+\)\s*(.*)', r'\+\s*(.*)'],
        'cons': [r'\(\-\)\s*(.*)', r'\-\s*(.*)']
    }
    
    items = []
    for pattern in patterns[pattern_type]:
        matches = re.findall(pattern, text)
        items.extend([clean_text(match) for match in matches])
    
    return items

def get_sentiment(text):
    """Get sentiment score and label using TextBlob with adjustable thresholds"""
    if pd.isna(text) or not text.strip():
        return 0.0, "neutral"
    
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity
    
    # Apply thresholds
    if score > SENTIMENT_THRESHOLDS['positive']:
        label = "positive"
    elif score < SENTIMENT_THRESHOLDS['negative']:
        label = "negative"
    else:
        label = "neutral"
    
    return score, label

def get_top_keywords(texts, top_n=5):
    """Extract top keywords using TF-IDF"""
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform the texts
    tfidf_matrix = tfidf.fit_transform(texts)
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Get top keywords for each document
    top_keywords_list = []
    for i in range(len(texts)):
        # Get TF-IDF scores for current document
        doc_scores = tfidf_matrix[i].toarray()[0]
        
        # Get indices of top scores
        top_indices = doc_scores.argsort()[-top_n:][::-1]
        
        # Get corresponding keywords
        top_keywords = [feature_names[idx] for idx in top_indices]
        top_keywords_list.append(top_keywords)
    
    return top_keywords_list

def classify_technical_terms(text_list):
    """Classify text into technical categories for each item in the list"""
    if not text_list:  # If list is empty
        return []
    
    # Process each item separately
    all_categories = []
    for item in text_list:
        item_categories = []
        for category, terms in TECH_CATEGORIES.items():
            if any(term.lower() in item.lower() for term in terms):
                item_categories.append(category)
        if item_categories:  # Only add if categories were found
            all_categories.append({
                'text': item,
                'categories': item_categories
            })
    
    return all_categories

def process_reviews():
    try:
        # Check if required packages are installed
        if not check_requirements():
            return

        # Download required NLTK data
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Check if input file exists
        input_file = 'review_data_full.xlsx'
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found!")
            return

        # Read the Excel file
        print("Reading Excel file...")
        df = pd.read_excel(input_file)
        print(f"Successfully read {len(df)} rows from {input_file}")
        
        # Clean and process pros and cons
        print("Processing pros and cons...")
        df['pros_cleaned'] = df['pros'].apply(lambda x: extract_pros_cons(x, 'pros'))
        df['cons_cleaned'] = df['cons'].apply(lambda x: extract_pros_cons(x, 'cons'))
        
        # Clean and process text
        print("Processing review text...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Perform sentiment analysis
        print("Performing sentiment analysis...")
        print(f"Using sentiment thresholds: {SENTIMENT_THRESHOLDS}")
        sentiment_results = df['cleaned_text'].apply(get_sentiment)
        df['sentiment_score'] = sentiment_results.apply(lambda x: x[0])
        df['sentiment_label'] = sentiment_results.apply(lambda x: x[1])
        
        # Extract top keywords
        print("Extracting keywords...")
        df['top_keywords'] = get_top_keywords(df['cleaned_text'].tolist())
        
        # Classify technical terms
        print("Classifying technical terms...")
        df['pros_tech_categories'] = df['pros_cleaned'].apply(classify_technical_terms)
        df['cons_tech_categories'] = df['cons_cleaned'].apply(classify_technical_terms)
        
        # Create separate columns for each category
        for category in TECH_CATEGORIES.keys():
            df[f'pros_{category}'] = df['pros_tech_categories'].apply(
                lambda x: [item['text'] for item in x if category in item['categories']]
            )
            df[f'cons_{category}'] = df['cons_tech_categories'].apply(
                lambda x: [item['text'] for item in x if category in item['categories']]
            )
        
        # Select and reorder columns for output
        output_columns = [
            'model', 'brand', 'segmentation', 'country',
            'pros_cleaned', 'cons_cleaned', 'cleaned_text',
            'sentiment_score', 'sentiment_label', 'top_keywords'
        ]
        
        # Add category-specific columns
        for category in TECH_CATEGORIES.keys():
            output_columns.extend([f'pros_{category}', f'cons_{category}'])
        
        # Create output DataFrame
        output_df = df[output_columns].copy()
        
        # Save to Excel
        output_file = 'processed_reviews.xlsx'
        print(f"Saving processed data to {output_file}...")
        output_df.to_excel(output_file, index=False)
        
        # Print summary
        print("\nProcessing complete!")
        print(f"Total reviews processed: {len(df)}")
        print("\nSentiment distribution:")
        print(df['sentiment_label'].value_counts())
        print("\nSentiment score statistics:")
        print(df['sentiment_score'].describe())
        
        # Print category statistics
        print("\nTechnical categories distribution:")
        for category in TECH_CATEGORIES.keys():
            print(f"\n{category.upper()}:")
            print(f"Pros mentions: {df[f'pros_{category}'].apply(len).sum()}")
            print(f"Cons mentions: {df[f'cons_{category}'].apply(len).sum()}")
        
        print("\nSample of processed data:")
        print(output_df.head())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    process_reviews() 