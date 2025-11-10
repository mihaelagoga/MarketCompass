import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from pytrends.request import TrendReq
from datetime import datetime
import time
import random
import numpy as np 
from dateutil.parser import parse as _parse
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


GCP_PROJECT_ID = 'your-gcp-project-id'

BIGQUERY_DATASET = 'market_data'

if GCP_PROJECT_ID == 'your-gcp-project-id':
    print("="*50)
    print("ERROR: Please update GCP_PROJECT_ID at the top of the script.")
    print("="*50)
    exit()

try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    dataset_ref = bq_client.dataset(BIGQUERY_DATASET)
    bq_client.get_dataset(dataset_ref)
    print(f"BigQuery dataset {BIGQUERY_DATASET} already exists.")
except NotFound:
    print(f"BigQuery dataset {BIGQUERY_DATASET} not found. Creating...")
    try:
        bq_client.create_dataset(dataset_ref, timeout=30)
        print(f"Dataset {BIGQUERY_DATASET} created.")
    except Exception as e:
        print(f"ERROR: Could not create dataset. {e}")
        exit()
except Exception as e:
    print(f"ERROR: Could not initialize BigQuery client. {e}")
    print("Please ensure you have authenticated with 'gcloud auth application-default login'")
    exit()


def create_table_if_not_exists(table_name, schema, partition_field):
    """
    Creates a BQ table with partitioning if it doesn't already exist.
    """
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{table_name}"
    
    try:
        bq_client.get_table(table_id)
    except NotFound:
        print(f"Table {table_id} not found. Creating...")
        try:
            table = bigquery.Table(table_id, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field,
            )
            bq_client.create_table(table)
            print(f"  Success. Table {table_id} created with partitioning on '{partition_field}'.")
        except Exception as e:
            print(f"  --- ERROR creating table {table_name} ---")
            print(f"  {e}")
            raise


def upload_to_bigquery(df, table_name, schema, partition_field='date'):
    """
    Uploads a DataFrame to BigQuery idempotently using a DELETE/INSERT pattern.
    - Ensures table exists (from create_table_if_not_exists helper).
    - Deletes all rows in the destination table that fall within the new data's date range.
    """
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{table_name}"

    try:
        create_table_if_not_exists(table_name, schema, partition_field)
    except Exception as e:
        print(f"Aborting upload to {table_name} due to table creation error.")
        return

    df_upload = df.copy()

    if df_upload.empty:
        print(f"No data provided for {table_name}, skipping upload.")
        return

    if partition_field not in df_upload.columns:
        print(f"ERROR: Partition field '{partition_field}' not in DataFrame columns: {df_upload.columns}")
        return
    df_upload[partition_field] = pd.to_datetime(df_upload[partition_field]).dt.strftime('%Y-%m-%d')

    for col_schema in schema:
        col_name = col_schema.name
        if col_name not in df_upload.columns:
            continue
        
        if col_schema.field_type == 'NUMERIC':
            df_upload[col_name] = pd.to_numeric(df_upload[col_name]).round(4)
        elif col_schema.field_type == 'INTEGER':
             df_upload[col_name] = df_upload[col_name].astype(float).astype('Int64')

    df_upload = df_upload.fillna(pd.NA).where(pd.notna(df_upload), None)
    df_upload = df_upload.replace({np.nan: None})

    min_date = df_upload[partition_field].min()
    max_date = df_upload[partition_field].max()

    if pd.isna(min_date) or pd.isna(max_date):
        print(f"  No valid dates found in data for {table_name}, skipping delete and insert.")
        return

    print(f"\nMaking upload idempotent: Deleting existing data in {table_name} between {min_date} and {max_date}...")
    
    try:
        query_params = [
            bigquery.ScalarQueryParameter("min_date", "DATE", min_date),
            bigquery.ScalarQueryParameter("max_date", "DATE", max_date),
        ]
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        
        delete_query = f"""
            DELETE FROM `{table_id}`
            WHERE {partition_field} BETWEEN @min_date AND @max_date
        """

        delete_job = bq_client.query(delete_query, job_config=job_config)
        delete_job.result()
        
        print(f"  Success. Deleted {delete_job.num_dml_affected_rows} existing rows.")

    except Exception as e:
        print(f"  --- ERROR during DELETE operation for {table_name} ---")
        print(f"  {e}")
        print(f"  Aborting upload for this table.")
        return

    rows_to_insert = df_upload.to_dict('records')

    if not rows_to_insert:
        print(f"  No data to upload for {table_name} after processing.")
        return

    print(f"  Uploading {len(rows_to_insert)} new rows to {table_id} (using streaming insert)...")
    
    try:
        errors = bq_client.insert_rows_json(table_id, rows_to_insert)
        
        if not errors:
            print(f"  Success. Streamed data.")
        else:
            print(f"  --- ERROR uploading {table_name} ---")
            print("  Errors encountered during streaming insert:")
            for i, error in enumerate(errors):
                if i < 20:
                    print(f"  - Row {error['index']}: {error['errors']}")
                elif i == 20:
                    print(f"  ... and {len(errors) - 20} more errors.")
                    
    except Exception as e:
        print(f"  --- FATAL ERROR uploading {table_name} ---")
        print(f"  {e}")
        print("  DataFrame Info:")
        df_upload.info()
        print("\n  DataFrame Head:")
        print(df_upload.head())

print("\nFetching FRED economic data...")
try:
    start_date = "2025-07-20"
    end_date = "2025-10-29"

    fred_data = pdr.get_data_fred(['FEDFUNDS', 'DGS10'], start=start_date, end=end_date)
    
    print("  FRED data was retrieved.")

    fred_schema = [
        bigquery.SchemaField('date', 'DATE'),
        bigquery.SchemaField('fed_Funds', 'NUMERIC'),
        bigquery.SchemaField('dgs_10', 'NUMERIC'),
    ]
    df_to_upload_fred = fred_data.reset_index()
    df_to_upload_fred.rename(columns={
        'DATE': 'date', 
        'FEDFUNDS': 'fed_Funds', 
        'DGS10': 'dgs_10'
    }, inplace=True)
    upload_to_bigquery(
        df_to_upload_fred, 
        'fred_economic_data', 
        fred_schema, 
        partition_field='date'
    )

except Exception as e:
    print(f"An error occurred fetching FRED data: {e}")


print("\nFetching S&P 500 and VIX data from Yahoo Finance...")
try:
    tickers = ["^GSPC", "^VIX"]
    start_date = "2025-07-20"
    end_date = "2025-10-29"

    market_data_raw = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)

    if not market_data_raw.empty:
        market_data = market_data_raw['Adj Close'].rename(columns={'^GSPC': 'SP500', '^VIX': 'VIX'})
        
        print("  Fetched market data.")

        market_schema = [
            bigquery.SchemaField('date', 'DATE'),
            bigquery.SchemaField('sp500', 'NUMERIC'),
            bigquery.SchemaField('vix', 'NUMERIC'),
        ]
        df_to_upload_market = market_data.reset_index()
        df_to_upload_market.rename(columns={
            'Date': 'date', 
            'SP500': 'sp500', 
            'VIX': 'vix'
        }, inplace=True)
        upload_to_bigquery(
            df_to_upload_market, 
            'market_data', 
            market_schema, 
            partition_field='date'
        )
        
    else:
        print("Error: No data was returned from Yahoo Finance.")

except Exception as e:
    print(f"An error occurred: {e}")

print("\nFetching Google Trends data...")

start_date = "2025-07-20"
end_date = "2025-10-29"

EARLIEST = pd.Timestamp("2004-01-01")
if pd.Timestamp(start_date) < EARLIEST:
    start_date = EARLIEST

start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
end_str   = pd.to_datetime(end_date).strftime("%Y-%m-%d")
timeframe = f"{start_str} {end_str}"
print("Using timeframe:", timeframe)

keywords = [
    "recession",
    "inflation",
    "unemployment",
    "stock market crash",
    "interest rates",
    "gold prices",
    "rare earth minerals"
]
ANCHOR = "recession"
if ANCHOR not in keywords:
    raise ValueError("ANCHOR must be in keywords")

MAX_PER_BATCH = 5

def make_batches(all_terms, anchor, max_size=MAX_PER_BATCH):

    others = [t for t in all_terms if t != anchor]
    batches = []
    i = 0
    while i < len(others):
        chunk = others[i:i + (max_size - 1)]
        batch = [anchor] + chunk
        batches.append(batch)
        i += (max_size - 1)
    if not batches:
        batches = [[anchor]]
    return batches

pytrends = TrendReq(hl='en-US', tz=0)

def fetch_batch_df(batch_terms, timeframe, geo='US', retries=4):
    attempt = 0
    backoff = 1.0
    last_exc = None
    while attempt < retries:
        try:
            print(f"Fetching batch (len={len(batch_terms)}): {batch_terms} timeframe={timeframe}")
            pytrends.build_payload(batch_terms, timeframe=timeframe, geo=geo, gprop='')
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                idx = pd.date_range(start=start_str, end=end_str, freq='D')
                df = pd.DataFrame({t: 0 for t in batch_terms}, index=idx)
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            df = df[[c for c in df.columns if c in batch_terms]]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            last_exc = e
            attempt += 1
            print(f"  Attempt {attempt}/{retries} failed: {type(e).__name__}: {e}")
            if any('400' in str(x) for x in e.args):
                print("HTTP 400 error")
                break
            time.sleep(backoff + random.random()*0.5)
            backoff = min(backoff * 2, 30.0)
    
    if last_exc:
        raise last_exc
    else:
        raise Exception("Failed to fetch batch data after retries.")


batches = make_batches(keywords, ANCHOR, MAX_PER_BATCH)
print("Batches to fetch:", batches)

batch_results = []
for b in batches:
    df_b = fetch_batch_df(b, timeframe=timeframe, geo='US', retries=5)
    batch_results.append((b, df_b))

ref_batch_terms, ref_df = batch_results[0]
combined = ref_df.copy()

for (batch_terms, df_batch) in batch_results[1:]:
    overlap_dates = sorted(set(combined.index.date).intersection(set(df_batch.index.date)))
    if len(overlap_dates) == 0:
        ref_max = combined[ANCHOR].max() if combined[ANCHOR].max() != 0 else 1.0
        cur_max = df_batch[ANCHOR].max() if df_batch[ANCHOR].max() != 0 else 1.0
        scale_anchor = ref_max / cur_max
        print(f"No overlap with combined; scaling batch by anchor max ratio {scale_anchor:.4f}")
    else:
        ratios = []
        for d in overlap_dates:
            val_ref = float(combined.loc[combined.index.date == d, ANCHOR].iloc[0])
            val_cur = float(df_batch.loc[df_batch.index.date == d, ANCHOR].iloc[0])
            if val_cur == 0:
                continue
            ratios.append(val_ref / val_cur)
        if len(ratios) == 0:
            ref_max = combined[ANCHOR].max() if combined[ANCHOR].max() != 0 else 1.0
            cur_max = df_batch[ANCHOR].max() if df_batch[ANCHOR].max() != 0 else 1.0
            scale_anchor = ref_max / cur_max
            print(f"Overlap present but anchor ratios invalid; fallback scale {scale_anchor:.4f}")
        else:
            scale_anchor = float(pd.Series(ratios).median())
            print(f"Scaling batch by median anchor ratio {scale_anchor:.6f} using {len(ratios)} overlap points")

    df_scaled = df_batch.copy().astype(float) * scale_anchor

    for col in df_scaled.columns:
        if col in combined.columns:
            missing_idx = df_scaled.index.difference(combined.index)
            if len(missing_idx) > 0:
                to_add = df_scaled.loc[missing_idx, [col]]
                combined = pd.concat([combined, to_add], axis=0)
        else:
            combined = pd.concat([combined, df_scaled[[col]]], axis=1)
    combined = combined.sort_index()

final = combined.loc[:, ~combined.columns.duplicated()]
for kw in keywords:
    if kw not in final.columns:
        final[kw] = float('nan')

final = final[keywords]
final = final.round(4).clip(lower=0, upper=100)
final.index = pd.to_datetime(final.index)

df_to_upload_trends = final.reset_index()

sanitized_cols = {'index': 'date'}
sanitized_keywords_list = []
for kw in keywords:
    sanitized_name = kw.replace(' ', '_').replace('-', '_')
    sanitized_cols[kw] = sanitized_name
    sanitized_keywords_list.append(sanitized_name)

df_to_upload_trends.rename(columns=sanitized_cols, inplace=True)

google_trends_schema = [bigquery.SchemaField('date', 'DATE')] + \
                     [bigquery.SchemaField(kw, 'NUMERIC') for kw in sanitized_keywords_list]

upload_to_bigquery(
    df_to_upload_trends, 
    'google_trends', 
    google_trends_schema, 
    partition_field='date'
)

print("\nGoogle Trends data processing and upload complete.")
print(final.head()) 
print(final.tail())

print("\n\nAll data fetching and uploading complete.")