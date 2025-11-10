import os
import json
import datetime 
from google.cloud import firestore


def json_serial_converter(o):
    """
    Custom JSON serializer for objects not serializable by default json code.
    Specifically handles datetime objects from Firestore.
    """
    if isinstance(o, datetime.datetime):
        return o.isoformat() 
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def download_firestore_collection():
    """
    Downloads all documents from the 'articles' collection in Firestore
    and saves them to a specific folder as a JSON file.
    """
   
    collection_name = "articles"
    output_file_path = "data/news_articles/raw_data/articles_export.json"

   
    print("Connecting to Firestore...")
    db = firestore.Client()

    print(f"Fetching all documents from the '{collection_name}' collection...")
    docs_ref = db.collection(collection_name).stream()

    all_articles = []
    doc_count = 0
    for doc in docs_ref:
        article_data = doc.to_dict()
        article_data['id'] = doc.id
        all_articles.append(article_data)
        doc_count += 1
        if doc_count % 100 == 0:
            print(f"  ...retrieved {doc_count} documents")

    print(f"\nSuccessfully retrieved a total of {doc_count} documents.")

    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)

    print(f"Saving data to '{output_file_path}'...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        
        json.dump(all_articles, f, ensure_ascii=False, indent=4, default=json_serial_converter)

    print("\nData has been exported.")


if __name__ == "__main__":
    download_firestore_collection()