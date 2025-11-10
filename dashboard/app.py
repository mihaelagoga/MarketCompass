import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.cloud import bigquery
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# --- Configuration & BigQuery Connection ---

# IMPORTANT: Authenticate with Google Cloud first.
# Run in your terminal: `gcloud auth application-default login`
# This code uses Application Default Credentials (ADC) to connect.

try:
    client = bigquery.Client()
    PROJECT_ID = client.project
    print(f"Successfully connected to BigQuery project: {PROJECT_ID}")
except Exception as e:
    print(f"Error connecting to BigQuery: {e}")
    print("Please ensure you have authenticated using `gcloud auth application-default login`")
    # Exit or raise if connection fails? For now, we'll let it fail at the query stage.

# Define your BigQuery table names
MASTER_DATA_TABLE = "`pivotal-glider-472219-r7.market_data.master_data`"
MODEL_RESULTS_TABLE = "`pivotal-glider-472219-r7.market_data.model_results`"

def load_data():
    """
    Loads data from the two BigQuery tables.
    """
    print("Loading data from BigQuery...")
    try:
        # Query Master Data
        sql_master = f"SELECT * FROM {MASTER_DATA_TABLE} ORDER BY date"
        df_master = client.query(sql_master).to_dataframe()
        
        # Query Model Results
        sql_models = f"SELECT * FROM {MODEL_RESULTS_TABLE} ORDER BY date"
        df_models = client.query(sql_models).to_dataframe()
        
        # --- Data Preprocessing ---
        
        # Convert date columns
        if 'date' in df_master.columns:
            df_master['date'] = pd.to_datetime(df_master['date'])
        if 'date' in df_models.columns:
            df_models['date'] = pd.to_datetime(df_models['date'])
            
        # Ensure numeric columns are numeric, coercing errors
        numeric_cols_master = [
            'news_volume', 'sentiment_interest_rates', 'sentiment_inflation', 
            'sentiment_unemployment', 'sentiment_stock_market', 'sentiment_recession', 
            'sentiment_tech_ai', 'sentiment_politics_regulation', 'sentiment_finance_banking', 
            'fed_Funds', 'dgs_10', 'recession', 'inflation', 'unemployment', 
            'stock_market_crash', 'interest_rates', 'gold_prices', 'rare_earth_minerals', 
            'sp500', 'vix'
        ]
        for col in numeric_cols_master:
            if col in df_master.columns:
                df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

        # Pre-map model advice for numeric plots (can be done once)
        advice_mapping = {'Buy': 1, 'Sell': 0}
        df_models_mapped = df_models.replace(advice_mapping).infer_objects(copy=False)
        df_models_mapped['ground_truth_value'] = pd.to_numeric(df_models_mapped['ground_truth_value'], errors='coerce')
        
        print("Data loaded and preprocessed successfully.")
        # Return both original and mapped model data
        return df_master, df_models, df_models_mapped
        
    except Exception as e:
        print(f"Failed to load data from BigQuery: {e}")
        # Return empty dataframes on failure to avoid app crash
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data on app startup
df_master, df_models, df_models_mapped = load_data()
models_to_compare = ['logit_advice', 'xgboost_advice', 'gru_advice']


# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server = app.server

# --- Reusable Components ---
def create_empty_figure(message="No data to display"):
    """Creates a blank figure with a text message."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": message, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
    )
    return fig

def create_kpi_card(title, value, subtitle):
    """Helper function to create a Bootstrap Card for KPIs."""
    return dbc.CardBody(
        [
            html.H5(title, className="card-title text-muted"),
            html.H3(value, className="card-text"),
            html.P(subtitle, className="card-text small text-muted"),
        ]
    )

# --- Tab 1: Market Data Layout ---
def create_market_data_tab():
    if df_master.empty:
        return html.Div([
            html.H4("Error: Could not load Market Data"),
            html.P("Please check your BigQuery connection and table names.")
        ], className="p-4")

    min_date = df_master['date'].min()
    max_date = df_master['date'].max()

    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Market Data Explorer", className="card-title"),
                html.P("Select a date range to filter all charts on all tabs."),
                dcc.DatePickerRange(
                    id='date-slider',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=min_date,
                    end_date=max_date,
                    display_format='YYYY-MM-DD'
                )
            ])
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(id='stock-market-chart')), width=12, lg=6, className="mb-4"),
            dbc.Col(dbc.Card(dcc.Graph(id='interest-rates-chart')), width=12, lg=6, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(id='commodities-chart')), width=12, lg=6, className="mb-4"),
            dbc.Col(dbc.Card(dcc.Graph(id='news-volume-chart')), width=12, lg=6, className="mb-4"),
        ]),
        dbc.Row([
            # Sentiment chart removed, gtrends chart is now full width
            dbc.Col(dbc.Card(dcc.Graph(id='gtrends-chart')), width=12, lg=12, className="mb-4"),
        ]),
    ], className="p-4")

# --- Tab 2: Sentiment Analysis Layout ---
def create_sentiment_analysis_tab():
    if df_master.empty:
        return html.Div([
            html.H4("Error: Could not load Market Data"),
            html.P("Please check your BigQuery connection and table names.")
        ], className="p-4")

    return html.Div([
        # Row for KPI Cards
        dbc.Row([
            dbc.Col(dbc.Card(id='avg-sentiment-card'), width=6, lg=3, className="mb-4"),
            dbc.Col(dbc.Card(id='pos-topic-card'), width=6, lg=3, className="mb-4"),
            dbc.Col(dbc.Card(id='neg-topic-card'), width=6, lg=3, className="mb-4"),
            dbc.Col(dbc.Card(id='vol-topic-card'), width=6, lg=3, className="mb-4"),
        ]),
        
        # Row for Daily Bar Chart
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(id='sentiment-bar-chart')), width=12, className="mb-4"),
        ]),
        
        # Row for Sentiment Trend Line Chart (Moved from Tab 1)
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(id='sentiment-chart')), width=12, className="mb-4"),
        ]),
        
        # Row for Pie Chart
        dbc.Row([
            dbc.Col(
                dbc.Card(dcc.Graph(id='sentiment-pie-chart')), 
                width=12, lg=8, className="mb-4 mx-auto" # Center the pie chart
            ),
        ]),
    ], className="p-4")

# --- Tab 3: Model Comparison Layout ---
def create_model_comparison_tab():
    # This function now just creates the layout. Logic is in the callback.
    if df_models.empty:
        return html.Div([
            html.H4("Error: Could not load Model Results"),
            html.P("Please check your BigQuery connection and table names.")
        ], className="p-4")

    # Create dynamic cards for each model
    model_cards = []
    for model in models_to_compare:
        model_name = model.split('_')[0].upper()
        model_cards.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(f"{model_name} Model"),
                    dbc.CardBody([
                        # Accuracy score will be populated by callback
                        html.H5(id=f"acc-{model}", className="card-title"),
                        # Confusion matrix figure will be populated by callback
                        dcc.Graph(id=f"cm-{model}", config={'displayModeBar': False})
                    ])
                ]),
                width=12, lg=4, className="mb-4"
            )
        )

    return html.Div([
        # Row for Accuracy & Confusion Matrices
        dbc.Row(model_cards, className="mb-4"),
        
        # New Row for Stacked Bar Chart
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(id='model-advice-barchart')), width=12, className="mb-4"),
        ]),
        
        # Row for Advice Over Time Line Chart
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(id='model-advice-linechart')), width=12, className="mb-4"),
        ]),
        
        # Row for Raw Data Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Raw Model Results (Filtered by Date)"),
                    dbc.CardBody(
                        dash_table.DataTable(
                            id='model-results-table', # ID to be updated by callback
                            columns=[{"name": i, "id": i} for i in df_models.columns],
                            data=df_models.to_dict('records'), # Initial data, will be updated
                            page_size=10,
                            sort_action="native",
                            filter_action="native",
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'minWidth': '100px', 'padding': '5px'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                        )
                    )
                ])
            ], width=12)
        ])
    ], className="p-4")


# --- App Layout ---
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="MarketCompass Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    # Tabs have been re-ordered
    dcc.Tabs(id="tabs-main", value='tab-1', children=[
        dcc.Tab(label='Market Data', value='tab-1', children=[create_market_data_tab()]),
        dcc.Tab(label='Sentiment Analysis', value='tab-2', children=[create_sentiment_analysis_tab()]), # Was Tab 3
        dcc.Tab(label='Model Comparison', value='tab-3', children=[create_model_comparison_tab()]), # Was Tab 2
    ]),
], fluid=True)


# --- Callbacks ---

@app.callback(
    [
        # Tab 1 (Market Data) Outputs - 5
        Output('stock-market-chart', 'figure'),
        Output('interest-rates-chart', 'figure'),
        Output('commodities-chart', 'figure'),
        Output('news-volume-chart', 'figure'),
        Output('gtrends-chart', 'figure'),
        
        # Tab 2 (Sentiment) Outputs - 7
        Output('sentiment-chart', 'figure'), 
        Output('avg-sentiment-card', 'children'),
        Output('pos-topic-card', 'children'),
        Output('neg-topic-card', 'children'),
        Output('vol-topic-card', 'children'),
        Output('sentiment-bar-chart', 'figure'),
        Output('sentiment-pie-chart', 'figure'),
        
        # Tab 3 (Model Comparison) Outputs - 9
        # CORRECTED ORDER: (acc, cm), (acc, cm), (acc, cm)
        Output('acc-logit_advice', 'children'),
        Output('cm-logit_advice', 'figure'),
        Output('acc-xgboost_advice', 'children'),
        Output('cm-xgboost_advice', 'figure'),
        Output('acc-gru_advice', 'children'),
        Output('cm-gru_advice', 'figure'),
        
        Output('model-advice-barchart', 'figure'),
        Output('model-advice-linechart', 'figure'),
        Output('model-results-table', 'data'),
    ],
    [
      # Main Input
     Input('date-slider', 'start_date'),
     Input('date-slider', 'end_date')
    ]
)
def update_all_charts(start_date, end_date):
    
    # --- Empty/Error Handling ---
    if df_master.empty or df_models.empty:
        empty_figs_t1 = [create_empty_figure("No Market Data")] * 5
        empty_cards_t2 = [create_kpi_card("No Data", "-", "-")] * 4
        empty_figs_t2 = [create_empty_figure("No Market Data")] * 3
        empty_text_t3 = ["Accuracy: N/A"] * 3
        empty_figs_t3_cm = [create_empty_figure("No Model Data")] * 3 # 3 CM figs
        empty_figs_t3_main = [create_empty_figure("No Model Data")] * 2 # 2 main figs
        empty_data_t3 = []
        
        # Assemble empty outputs in the new correct order
        empty_t3_metrics = [item for pair in zip(empty_text_t3, empty_figs_t3_cm) for item in pair]
        
        return (
            *empty_figs_t1, 
            empty_figs_t2[0], *empty_cards_t2, *empty_figs_t2[1:], 
            *empty_t3_metrics, *empty_figs_t3_main, empty_data_t3
        )

    # --- Filter Dataframes ---
    dff_master = df_master[(df_master['date'] >= start_date) & (df_master['date'] <= end_date)]
    dff_models = df_models[(df_models['date'] >= start_date) & (df_models['date'] <= end_date)]
    dff_models_mapped = df_models_mapped[(df_models_mapped['date'] >= start_date) & (df_models_mapped['date'] <= end_date)]

    if dff_master.empty:
        # Still return empty for all, as master data is used for sentiment
        empty_figs_t1 = [create_empty_figure("No data in selected range")] * 5
        empty_cards_t2 = [create_kpi_card("No Data", "-", "-")] * 4
        empty_figs_t2 = [create_empty_figure("No data in selected range")] * 3
        empty_text_t3 = ["Accuracy: N/A"] * 3
        empty_figs_t3_cm = [create_empty_figure("No Model Data in Range")] * 3
        empty_figs_t3_main = [create_empty_figure("No Model Data in Range")] * 2
        empty_data_t3 = []
        
        # Assemble empty outputs in the new correct order
        empty_t3_metrics = [item for pair in zip(empty_text_t3, empty_figs_t3_cm) for item in pair]
        
        return (
            *empty_figs_t1, 
            empty_figs_t2[0], *empty_cards_t2, *empty_figs_t2[1:], 
            *empty_t3_metrics, *empty_figs_t3_main, empty_data_t3
        )

    # --- Tab 1: Market Data Charts ---
    
    # Chart 1: Stock Market
    fig_stock = make_subplots(specs=[[{"secondary_y": True}]])
    fig_stock.add_trace(go.Scatter(x=dff_master['date'], y=dff_master['sp500'], name='S&P 500', line=dict(color='blue'), connectgaps=True), secondary_y=False)
    fig_stock.add_trace(go.Scatter(x=dff_master['date'], y=dff_master['vix'], name='VIX', line=dict(color='orange'), connectgaps=True), secondary_y=True)
    fig_stock.update_layout(title_text="Stock Market (S&P 500 vs. VIX)", hovermode="x unified")
    fig_stock.update_yaxes(title_text="S&P 500", secondary_y=False)
    fig_stock.update_yaxes(title_text="VIX", secondary_y=True)

    # Chart 2: Interest Rates
    fig_rates = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rates.add_trace(go.Scatter(x=dff_master['date'], y=dff_master['fed_Funds'], name='Fed Funds', line=dict(color='green'), connectgaps=True), secondary_y=False)
    fig_rates.add_trace(go.Scatter(x=dff_master['date'], y=dff_master['dgs_10'], name='10-Yr Treasury (DGS10)', line=dict(color='red'), connectgaps=True), secondary_y=True)
    fig_rates.update_layout(title_text="Interest Rates (Fed Funds vs. 10-Yr)", hovermode="x unified")
    fig_rates.update_yaxes(title_text="Fed Funds Rate", secondary_y=False)
    fig_rates.update_yaxes(title_text="DGS10", secondary_y=True)

    # Chart 3: Commodities
    fig_comm = make_subplots(specs=[[{"secondary_y": True}]])
    fig_comm.add_trace(go.Scatter(x=dff_master['date'], y=dff_master['gold_prices'], name='Gold Prices', line=dict(color='gold'), connectgaps=True), secondary_y=False)
    fig_comm.add_trace(go.Scatter(x=dff_master['date'], y=dff_master['rare_earth_minerals'], name='Rare Earths', line=dict(color='purple'), connectgaps=True), secondary_y=True)
    fig_comm.update_layout(title_text="Commodities (Gold vs. Rare Earths)", hovermode="x unified")
    fig_comm.update_yaxes(title_text="Gold Prices", secondary_y=False)
    fig_comm.update_yaxes(title_text="Rare Earths Index", secondary_y=True)

    # Chart 4: News Volume
    fig_news = px.line(dff_master, x='date', y='news_volume', title='News Volume Over Time')
    fig_news.update_layout(hovermode="x unified")

    # Chart 5: Google Trends
    gtrends_cols = ['recession', 'inflation', 'unemployment', 'stock_market_crash', 'interest_rates']
    fig_gtrends = px.line(dff_master, x='date', y=gtrends_cols, title='Google Trends Interest')
    fig_gtrends.update_layout(hovermode="x unified", legend_title="Search Term")
    
    tab1_outputs = (fig_stock, fig_rates, fig_comm, fig_news, fig_gtrends)

    # --- Tab 2 (Sentiment Analysis) ---
    
    sentiment_cols = [col for col in dff_master.columns if col.startswith('sentiment_')]
    
    # Chart 6: Sentiment Scores (Line) - This is for the chart now on Tab 2
    fig_sentiment_line = px.line(dff_master, x='date', y=sentiment_cols, title='News Sentiment Scores (Trend)')
    fig_sentiment_line.update_layout(hovermode="x unified", legend_title="Sentiment Topic")
    
    if not sentiment_cols or dff_master[sentiment_cols].empty:
        # Handle case where no sentiment columns exist
        avg_card_body = create_kpi_card("Avg. Sentiment", "N/A", "No sentiment data")
        pos_card_body = create_kpi_card("Most Positive", "N/A", "No sentiment data")
        neg_card_body = create_kpi_card("Most Negative", "N/A", "No sentiment data")
        vol_card_body = create_kpi_card("Most Volatile", "N/A", "No sentiment data")
        fig_sentiment_bar = create_empty_figure("No sentiment data")
        fig_sentiment_pie = create_empty_figure("No sentiment data")
    
    else:
        # KPI Card Calculations
        dff_sentiments = dff_master[sentiment_cols]
        avg_overall = dff_sentiments.mean().mean() # Avg of all scores
        avg_by_topic = dff_sentiments.mean()
        std_by_topic = dff_sentiments.std()
        
        most_pos_topic = avg_by_topic.idxmax()
        most_pos_val = avg_by_topic.max()
        
        most_neg_topic = avg_by_topic.idxmin()
        most_neg_val = avg_by_topic.min()
        
        most_vol_topic = std_by_topic.idxmax()
        most_vol_val = std_by_topic.max()

        # Create Card Bodies
        avg_card_body = create_kpi_card("Avg. Overall Sentiment", f"{avg_overall:.3f}", "Across all topics")
        pos_card_body = create_kpi_card("Most Positive Topic", most_pos_topic.replace('sentiment_', '').title(), f"Avg: {most_pos_val:.3f}")
        neg_card_body = create_kpi_card("Most Negative Topic", most_neg_topic.replace('sentiment_', '').title(), f"Avg: {most_neg_val:.3f}")
        vol_card_body = create_kpi_card("Most Volatile Topic", most_vol_topic.replace('sentiment_', '').title(), f"Std. Dev: {most_vol_val:.3f}")

        # Sentiment Bar Chart (Daily)
        dff_melted = dff_master.melt(id_vars=['date'], value_vars=sentiment_cols, var_name='Sentiment Topic', value_name='Sentiment Score')
        fig_sentiment_bar = px.bar(dff_melted, x='date', y='Sentiment Score', color='Sentiment Topic', 
                                   title='Daily Sentiment Scores by Topic', barmode='group')
        fig_sentiment_bar.update_layout(hovermode="x unified")

        # Sentiment Pie Chart (Share of Positive Sentiment)
        positive_topics = avg_by_topic[avg_by_topic > 0]
        if positive_topics.empty:
            fig_sentiment_pie = create_empty_figure("No positive sentiment topics in this period")
        else:
            pie_names = [n.replace('sentiment_', '').title() for n in positive_topics.index]
            fig_sentiment_pie = px.pie(values=positive_topics.values, 
                                       names=pie_names, 
                                       title='Share of Positive Sentiment by Topic (Period Average)')
            fig_sentiment_pie.update_traces(textposition='inside', textinfo='percent+label')

    tab2_outputs = (
        fig_sentiment_line, avg_card_body, pos_card_body, neg_card_body, 
        vol_card_body, fig_sentiment_bar, fig_sentiment_pie
    )

    # --- Tab 3 (Model Comparison) ---
    
    # Check if filtered model data is empty
    if dff_models.empty:
        empty_text_t3 = ["Accuracy: N/A"] * 3
        empty_figs_t3_cm = [create_empty_figure("No Model Data in Range")] * 3
        empty_figs_t3_main = [create_empty_figure("No Model Data in Range")] * 2
        empty_data_t3 = []
        
        # Assemble empty outputs in the new correct order
        model_metric_outputs = [item for pair in zip(empty_text_t3, empty_figs_t3_cm) for item in pair]
        
        tab3_outputs = (*model_metric_outputs, *empty_figs_t3_main, empty_data_t3)
        return (*tab1_outputs, *tab2_outputs, *tab3_outputs)

    # --- Calculate Metrics for each model ---
    labels = ['Sell', 'Buy']
    model_metric_outputs = [] # CORRECTED: Use one list in the correct order
    
    for model in models_to_compare:
        temp_df = dff_models[['ground_truth_advice', model]].copy()
        valid_labels = set(labels)
        mask = (
            temp_df['ground_truth_advice'].isin(valid_labels) &
            temp_df[model].isin(valid_labels)
        )
        filtered_df = temp_df[mask]
        y_true_filtered = filtered_df['ground_truth_advice']
        y_pred_filtered = filtered_df[model]
        
        if y_true_filtered.empty or y_pred_filtered.empty:
            acc = np.nan
            cm = np.zeros((len(labels), len(labels)))
        else:
            acc = accuracy_score(y_true_filtered, y_pred_filtered)
            cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
        
        # Accuracy Text
        acc_text = f"Accuracy: {acc:.2%}" if not np.isnan(acc) else "Accuracy: N/A"
        
        # Confusion Matrix Figure
        cm_fig = px.imshow(cm,
                           labels=dict(x="Predicted Advice", y="True Advice", color="Count"),
                           x=labels,
                           y=labels,
                           text_auto=True,
                           color_continuous_scale='Blues'
                          )
        cm_fig.update_layout(title=f"{model.split('_')[0].upper()} Confusion Matrix")
        
        # Append in the correct (acc, cm) order
        model_metric_outputs.append(acc_text)
        model_metric_outputs.append(cm_fig)

    # --- New Stacked Bar Chart (Daily Timeline) ---
    bar_chart_cols = ['ground_truth_advice'] + models_to_compare
    dff_melted_models = dff_models.melt(id_vars=['date'], value_vars=bar_chart_cols, var_name='Model', value_name='Advice')
    
    # Clean up model names for the chart
    model_name_map = {
        'ground_truth_advice': 'Ground Truth',
        'logit_advice': 'Logit',
        'xgboost_advice': 'XGBoost',
        'gru_advice': 'GRU'
    }
    dff_melted_models['Model'] = dff_melted_models['Model'].map(model_name_map)
    
    # Map Advice to numerical values for stacking
    dff_melted_models['value'] = dff_melted_models['Advice'].map({'Buy': 1, 'Sell': -1})
    
    # Filter out any rows with missing advice (e.g., NaN from mapping)
    dff_melted_models = dff_melted_models.dropna(subset=['value'])

    if dff_melted_models.empty:
        fig_model_barchart = create_empty_figure("No advice data in range")
    else:
        fig_model_barchart = px.bar(dff_melted_models, 
                                    x='date', 
                                    y='value', 
                                    color='Model', # Color by model/source
                                    title='Daily Model Advice vs. Ground Truth', 
                                    barmode='stack', # Stack bars
                                    )
        
        # Update layout
        fig_model_barchart.update_layout(
            yaxis_title="Net Advice",
            legend_title="Model"
        )
        # Add a horizontal line at y=0
        fig_model_barchart.add_hline(y=0, line_width=2, line_dash="dash", line_color="black")
    
    # --- Advice Over Time Line Chart ---
    plot_cols = ['ground_truth_value'] + models_to_compare
    # Use the filtered mapped dataframe
    valid_cols = [col for col in plot_cols if col in dff_models_mapped.columns and pd.api.types.is_numeric_dtype(dff_models_mapped[col])]
    
    if valid_cols and not dff_models_mapped.empty:
        fig_advice_line = px.line(dff_models_mapped, x='date', y=valid_cols,
                             title="Model Advice vs. Ground Truth")
        fig_advice_line.update_traces(line_shape='hv')
        fig_advice_line.update_layout(yaxis_title="Advice", legend_title="Model")
    else:
        fig_advice_line = create_empty_figure("No valid model advice columns to plot.")

    # --- Filtered Table Data ---
    table_data = dff_models.to_dict('records')

    # --- Assemble Tab 3 Outputs ---
    tab3_outputs = (
        *model_metric_outputs, # This now contains [acc1, cm1, acc2, cm2, acc3, cm3]
        fig_model_barchart,
        fig_advice_line,
        table_data
    )
    
    # --- Return All Outputs ---
    return (*tab1_outputs, *tab2_outputs, *tab3_outputs)


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)

