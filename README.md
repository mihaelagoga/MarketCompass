# MarketCompass

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MarketCompass is a comprehensive data analysis and visualization tool designed to provide insights into market trends, news sentiment, and economic indicators. It extracts data from various sources, processes it through machine learning models, and presents the findings in an interactive dashboard.

## Features

- **Interactive Dashboard:** A web-based dashboard built with Plotly Dash to visualize market data, sentiment analysis, and model predictions.
- **Data Aggregation:** Fetches data from multiple sources, including:
    - **Market Data:** S&P 500 and VIX from Yahoo Finance.
    - **Economic Indicators:** Federal Funds Rate and 10-Year Treasury rates from FRED.
    - **News Articles:** Financial news from various sources via NewsAPI.
    - **Google Trends:** Interest over time for key economic terms.
- **Sentiment Analysis:** Analyzes the sentiment of news articles to gauge market mood using a custom-trained model.
- **Predictive Modeling:** Utilizes Logistic Regression, XGBoost, and a GRU neural network to predict market trends.
- **Cloud Integration:** Uses Google Cloud Platform for data storage (BigQuery, Firestore) and serverless functions.

## Architecture

The project follows a multi-stage data pipeline:

1.  **News Data Collection:**
    -   A Google Cloud Function (`cloud_functions/fetch_news`) is triggered periodically (e.g., daily) to fetch news articles from the NewsAPI.
    -   The fetched articles are stored in a Firestore collection named `articles`.

2.  **Market Data Collection:**
    -   The `auxiliary_functions/fetch_market_data.py` script is run to collect market data (S&P 500, VIX), economic indicators (interest rates), and Google Trends data.
    -   This data is uploaded to several tables in a BigQuery dataset named `market_data`.

3.  **Sentiment Analysis Pipeline:**
    -   `auxiliary_functions/download_from_firestore.py` downloads the news articles from Firestore into a local JSON file (`articles_export.json`).
    -   The `sentiment_analysis/sentiment_model_training_pipeline.ipynb` notebook is used to train a custom sentiment analysis model. The trained model is saved locally.
    -   The `market_compass_modelling/run_sentiment_pipeline.ipynb` notebook uses the trained sentiment model to perform aspect-based sentiment analysis on the news articles and uploads the aggregated daily sentiment scores to a BigQuery table named `daily_sentiment_absa`.

4.  **Predictive Modeling Pipeline:**
    -   The `market_compass_modelling/model_2_training_pipeline.ipynb` notebook preprocesses the data from BigQuery and creates training, testing, and validation datasets (`train_data.csv`, `test_data.csv`, `validation_data.csv`).
    -   The `market_compass_modelling/run_model2_pipeline.ipynb` notebook trains the predictive models (Logistic Regression, XGBoost, GRU) on the prepared data, evaluates their performance, and saves the trained models. It then runs predictions and uploads the results to a BigQuery table named `model_results`.

5.  **Data Visualization:**
    -   The Dash application (`dashboard/app.py`) queries the `master_data` and `model_results` tables from BigQuery.
    -   It presents the data in an interactive dashboard with tabs for market data, sentiment analysis, and model comparison.

## Getting Started

### Prerequisites

-   Python 3.8+
-   Google Cloud SDK installed and authenticated.
-   A Google Cloud Platform project with the following APIs enabled:
    -   BigQuery API
    -   Firestore API
-   A NewsAPI key.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MarketCompass.git
    cd MarketCompass
    ```

2.  **Set up environment variables:**
    -   Create a `.env` file in the root of the `MarketCompass` directory.
    -   Add your NewsAPI key to the `.env` file:
        ```
        NEWS_API_KEY=your_news_api_key
        ```

3.  **Install dependencies:**
    -   **Dashboard:**
        ```bash
        pip install -r dashboard/requirements.txt
        ```
    -   **Cloud Function:**
        ```bash
        pip install -r cloud_functions/fetch_news/requirements.txt
        ```
    -   **Auxiliary and Modeling Scripts:**
        ```bash
        pip install pandas yfinance pandas_datareader pytrends google-cloud-bigquery scikit-learn xgboost torch transformers datasets
        ```

## Usage

The following steps should be followed in order to run the project.

### 1. Data Collection

1.  **Deploy the News Fetching Cloud Function:**
    -   Deploy the `fetch_news` Google Cloud Function. You can find instructions in the [Google Cloud documentation](https://cloud.google.com/functions/docs/deploying/python).
    -   Set up a Cloud Scheduler job to trigger the function daily to keep your news data up to date.

2.  **Fetch Market Data:**
    -   Update the `GCP_PROJECT_ID` in `auxiliary_functions/fetch_market_data.py`.
    -   Run the script to populate your BigQuery tables:
        ```bash
        python auxiliary_functions/fetch_market_data.py
        ```

### 2. Sentiment Analysis

1.  **Download News Articles:**
    -   Run the `download_from_firestore.py` script to download the news articles from Firestore.
        ```bash
        python auxiliary_functions/download_from_firestore.py
        ```

2.  **Train the Sentiment Model:**
    -   Open and run the `sentiment_analysis/sentiment_model_training_pipeline.ipynb` notebook to train the sentiment analysis model.

3.  **Run the Sentiment Pipeline:**
    -   Open and run the `market_compass_modelling/run_sentiment_pipeline.ipynb` notebook to generate and upload the sentiment scores.

### 3. Predictive Modeling

1.  **Prepare Modeling Data:**
    -   Open and run the `market_compass_modelling/model_2_training_pipeline.ipynb` notebook to create the training, testing, and validation datasets.

2.  **Train and Run Models:**
    -   Open and run the `market_compass_modelling/run_model2_pipeline.ipynb` notebook to train the predictive models, save them, and upload their predictions to BigQuery.

### 4. Running the Dashboard

1.  **Authenticate with Google Cloud:**
    ```bash
    gcloud auth application-default login
    ```

2.  **Run the dashboard application:**
    ```bash
    cd dashboard
    python app.py
    ```
    The dashboard will be available at `http://127.0.0.1:8050/`.

## Project Structure

```
MarketCompass/
├── .env                      # Environment variables (e.g., API keys)
├── auxiliary_functions/      # Scripts for data collection and processing
│   ├── analyze_ngrams.py       # Analyzes n-grams from the news data
│   ├── download_from_firestore.py # Downloads news articles from Firestore
│   └── fetch_market_data.py    # Fetches market data and uploads to BigQuery
├── cloud_functions/          # Serverless functions for data collection
│   └── fetch_news/             # Fetches news from NewsAPI and stores in Firestore
├── dashboard/                # The Dash web application
│   ├── app.py                  # The main dashboard application code
│   ├── Dockerfile              # Dockerfile for containerizing the dashboard
│   └── requirements.txt        # Python dependencies for the dashboard
├── market_compass_modelling/ # Notebooks for the predictive modeling pipeline
│   ├── model_2_training_pipeline.ipynb # Prepares data for the predictive models
│   ├── run_model2_pipeline.ipynb       # Trains and runs the predictive models
│   └── run_sentiment_pipeline.ipynb  # Runs the sentiment analysis pipeline
└── sentiment_analysis/       # Notebooks for the sentiment analysis model
    ├── sentiment_model_training_pipeline.ipynb # Trains the sentiment model
    └── sentiment_model.ipynb         # Defines the sentiment model architecture
```

## Data

The project uses several data files that are either generated by the scripts or used for training:

-   `1_unigrams.csv`, `2_bigrams.csv`, `3_trigrams.csv`: These files are generated by `analyze_ngrams.py` and contain the frequency of n-grams in the news articles.
-   `train_data.csv`, `test_data.csv`, `validation_data.csv`: These files are generated by `model_2_training_pipeline.ipynb` and are used to train and evaluate the predictive models.
-   `data/news_articles/raw_data/articles_export.json`: This file is generated by `download_from_firestore.py` and contains the raw news articles downloaded from Firestore.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.