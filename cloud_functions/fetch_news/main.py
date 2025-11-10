import os
import requests
import hashlib
from google.cloud import firestore
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)


def fetch_daily_news(event, context):
    """
    A Google Cloud Function to be triggered daily. It fetches news from the
    last 24 hours and stores them in Firestore.
    """
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
    if not NEWS_API_KEY:
           print("FATAL: NEWS_API_KEY environment variable not set.")
           return "API Key not found.", 500

    db = firestore.Client()
    url = 'https://newsapi.org/v2/everything'
    
    now_utc = datetime.now(timezone.utc)
    start_time_utc = now_utc - timedelta(hours=730)
    from_timestamp = start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

    print(f"Function triggered. Fetching new articles published since {from_timestamp} UTC.")
    
    quality_domains = 'bloomberg.com,wsj.com,reuters.com,cnbc.com,businessinsider.com'
    queries = {
        "market_concepts": '"market trends" OR "wall street" OR "capital gain" OR "NYSE" OR "NASDAQ" OR "investment" OR "economy" OR "financial market"',
        "major_indices": '"S&P 500" OR "Dow Jones"',
        "major_companies": '"Vanguard" OR "BlackRock" OR "Goldman Sachs" OR "AAPL" OR "MSFT" OR "GOOGL" OR "AMZN" OR "TSLA" OR "META"',
        "popular_etfs": '"VWCE" OR "VUSA" OR "VUSD" OR "VWRL" OR "VOO" OR "SPY"'
    }

    all_articles = {}
    
    try:
        for category, query in queries.items():
            print(f"--- Fetching news for category: {category} ---")
            
            params = {
                'q': query, 'domains': quality_domains, 'searchIn': 'title,description',
                'from': from_timestamp, 'language': 'en', 'sortBy': 'publishedAt',
                'pageSize': 100, 'apiKey': NEWS_API_KEY
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            articles_in_response = data.get('articles', [])
            print(f"Found {len(articles_in_response)} articles for this category.")

            for article in articles_in_response:
                article_url = article.get('url')
                if article_url: all_articles[article_url] = article

        if not all_articles:
            print ("No new articles found across all categories in the last 24 hours.")
            return "No new articles found.", 200
        
        print(f"\nFetched a total of {len(all_articles)} unique articles. Storing in Firestore...")
        saved_count = 0

        for article_url, article in all_articles.items():
            published_at = article.get('publishedAt')
            title = article.get('title')

            if published_at and title:
                unique_string = f"{article_url}{title}{published_at}"
                doc_id = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

                article_data = {
                    'article_content': article.get('content'), 'source': article_url,
                    'source_name': article.get('source', {}).get('name'),
                    'published_at': published_at, 'uploaded_at': firestore.SERVER_TIMESTAMP
                }
                
                doc_ref = db.collection('articles').document(doc_id)
                doc_ref.set(article_data)
                saved_count += 1

        print (f"Successfully saved {saved_count} news articles to Firestore.")
        return f"Successfully saved {saved_count} news articles to Firestore.", 200

    except requests.exceptions.RequestException as e:
        print(f"Error calling NewsAPI: {e}")
        return f"Error from external source: {e}", 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}", 500

if __name__ == '__main__':
    fetch_daily_news(None, None)