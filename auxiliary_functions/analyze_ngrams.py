import json
import re
import csv
import nltk
import argparse
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' tokenizer data...")
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' data...")
    nltk.download('stopwords', quiet=True)

def preprocess_text(text, stop_words):
    """
    Cleans and tokenizes text.
    1. Lowercases
    2. Removes punctuation and numbers (keeps only alphabetic words)
    3. Tokenizes
    4. Removes stop words
    """

    text = text.lower()
    
    tokens = re.findall(r'\b[a-z]+\b', text)
    
    processed_tokens = [
        word for word in tokens if word not in stop_words
    ]
    return processed_tokens

def write_counts_to_csv(counts, filename):
    print(f"Writing {filename}...")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow(['ngram', 'count'])

        for ngram, count in counts.most_common():
            writer.writerow([ngram, count])
    print(f"Successfully wrote {len(counts)} n-grams to {filename}.")

def main():
    """
    Main function to run the n-gram analysis.
    """
    
    parser = argparse.ArgumentParser(
        description="Run n-gram analysis on a JSON file and output CSVs."
    )
    parser.add_argument(
        "json_input_file", 
        type=str, 
        help="The path to the input JSON file."
    )
    args = parser.parse_args()
    
    json_input_file = args.json_input_file

    output_files = {
        1: '1_unigrams.csv',
        2: '2_bigrams.csv',
        3: '3_trigrams.csv'
    }

    stop_words = set(stopwords.words('english'))

    custom_stop_words = {'said', 'also', 'company', 'would', 'could', 'year'}
    stop_words.update(custom_stop_words)

    all_processed_tokens = []

    print(f"Loading and processing {json_input_file}...")
    try:
        with open(json_input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for article in data:
            if 'article_content' in article:
                content = article['article_content']
                tokens = preprocess_text(content, stop_words)
                all_processed_tokens.extend(tokens)
            else:
                print("Warning: Found an item in JSON without 'article_content' key.")
                
    except FileNotFoundError:
        print(f"Error: The file '{json_input_file}' was not found.")
        print("Please check the file path and try again.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_input_file}'.")
        print("Please check that the file is a valid JSON.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    if not all_processed_tokens:
        print("No processable text was found. Exiting.")
        return
        
    print(f"Processed a total of {len(all_processed_tokens)} tokens from {len(data)} articles.")

    print("\nGenerating and counting n-grams...")
    
    unigrams_gen = ngrams(all_processed_tokens, 1)
    bigrams_gen = ngrams(all_processed_tokens, 2)
    trigrams_gen = ngrams(all_processed_tokens, 3)
    
    unigram_counts = Counter([' '.join(gram) for gram in unigrams_gen])
    bigram_counts = Counter([' '.join(gram) for gram in bigrams_gen])
    trigram_counts = Counter([' '.join(gram) for gram in trigrams_gen])

    write_counts_to_csv(unigram_counts, output_files[1])
    write_counts_to_csv(bigram_counts, output_files[2])
    write_counts_to_csv(trigram_counts, output_files[3])
    
    print("\nAnalysis complete. All files saved.")

if __name__ == "__main__":
    main()