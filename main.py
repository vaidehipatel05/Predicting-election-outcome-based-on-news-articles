import dash
from dash import html, dcc, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from textblob import TextBlob
from lime import lime_tabular
from pybloom_live import BloomFilter
import dash_html_components as html
import boto3
from io import StringIO
from creds import ACCESS_KEY, SECRET_KEY

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and preprocess data
#df = pd.read_csv('./updated_dataset.csv')
session = boto3.Session(aws_access_key_id=ACCESS_KEY,
aws_secret_access_key=SECRET_KEY)

# S3 bucket name
bucket_name = 'kafka-news-election-project-bd'

# List to store DataFrames
dfs = []

# Create an S3 client
s3_client = session.client('s3')

# List objects in the bucket
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='news_market_')

# Iterate over each object in the bucket
for obj in response['Contents']:
    file_key = obj['Key']
    if file_key.endswith('.csv'):
        # Get the CSV file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(response['Body'])
        
        # Append the DataFrame to the list
        dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
df = combined_df
df.fillna('', inplace=True)

# NLTK setup
nltk.download(['stopwords', 'wordnet', 'punkt'])
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Bloom Filter Setup
bloom = BloomFilter(capacity=10000, error_rate=0.01)

# Text preprocessing with Bloom filter check
def preprocess_text(text):
    if text in bloom:
        return None  # Skip processing if text already processed
    else:
        bloom.add(text)
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and not word in stop_words])

df['clean_text'] = df['title'] + ' ' + df['fetched_text']
df['processed_text'] = df['clean_text'].apply(preprocess_text)
df.dropna(subset=['processed_text'], inplace=True)

# Feature extraction for LDA
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_counts = vectorizer.fit_transform(df['processed_text'])

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_counts)
df['cluster'] = clusters
cluster_labels = {0: 'Republican', 1: 'Democratic'}
df['predicted_label'] = df['cluster'].map(cluster_labels)

# Train a classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_counts, df['cluster'])

# Initialize LIME for tabular data
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_counts.toarray(),
    feature_names=vectorizer.get_feature_names_out(),
    class_names=['Republican', 'Democratic'],
    mode='classification',
    discretize_continuous=True
)

# LDA Analysis
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_counts)

# Sentiment Analysis
df['sentiment'] = df['processed_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment_category'] = pd.cut(df['sentiment'], bins=[-1, -0.01, 0.01, 1], labels=['negative', 'neutral', 'positive'])

# Applying k-anonymity
def apply_k_anonymity(df, k=3):
    freq = df['processed_text'].value_counts()
    to_keep = freq[freq >= k].index
    return df[df['processed_text'].isin(to_keep)]

df_k_anon = apply_k_anonymity(df, k=3)

# App layout
app.layout = html.Div([
    html.H1("US Elections News Analysis Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='Data Overview', children=[
            html.Div([
                html.H3("Data Overview"),
                dash_table.DataTable(
                    data=df.head().to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    page_size=10,
                ),
            ]),
        ]),
        dcc.Tab(label='Topic Modeling', children=[
            html.Div([
                html.H3("LDA Topic Modeling"),
                dcc.Graph(id='lda-topics')
            ]),
        ]),
        dcc.Tab(label='Sentiment Analysis', children=[
            html.Div([
                html.H3("Sentiment Analysis"),
                dcc.Graph(id='sentiment-analysis')
            ]),
        ]),
        dcc.Tab(label='Election Prediction', children=[
            html.Div([
                html.H3("Election Outcome Prediction"),
                dcc.Graph(id='election-prediction')
            ]),
        ]),
        dcc.Tab(label='LIME Explanation', children=[
            html.Div([
                html.H3("LIME Explanation for a Sample Prediction"),
                dcc.Graph(id='lime-explanation-graph')
            ]),
        ]),
        dcc.Tab(label='K-Anonymity Impact', children=[
            html.Div([
                html.H3("Impact of Applying K-Anonymity"),
                dcc.Graph(id='k-anonymity-impact')
            ]),
        ])
    ])
])

# Callbacks for updating the graphs
@app.callback(
    Output('lda-topics', 'figure'),
    Input('lda-topics', 'id')
)
def update_lda_topics(_):
    topic_names = ["Economy", "Healthcare", "Foreign Policy", "Education", "Environment"]
    fig = go.Figure()
    for i, topic in enumerate(lda.components_):
        top_features = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        fig.add_trace(go.Bar(x=top_features, y=topic[-10:], name=f"Topic {i+1}: {topic_names[i]}"))
    return fig

@app.callback(
    Output('sentiment-analysis', 'figure'),
    Input('sentiment-analysis', 'id')
)
def update_sentiment_analysis(_):
    # Check if the necessary columns exist and are not empty
    if 'predicted_label' in df.columns and 'sentiment_category' in df.columns and not df['predicted_label'].empty and not df['sentiment_category'].empty:
        # Try to create the bar chart
        try:
            # Grouping data to ensure it's aggregated properly for plotting
            sentiment_counts = df.groupby(['predicted_label', 'sentiment_category']).size().reset_index(name='counts')
            # Plotting the grouped data
            fig = px.bar(sentiment_counts, x='predicted_label', y='counts', color='sentiment_category',
                         labels={'predicted_label': 'Political Category', 'counts': 'Number of Articles'},
                         title='Sentiment Analysis by Political Category')
            # Adjusting bar width
            fig.update_traces(width=0.2)  # Adjust bar width as needed
            return fig
        except Exception as e:
            print("Error creating sentiment analysis chart:", str(e))
            # Return an empty figure if there is an error
            return go.Figure()
    else:
        print("Necessary columns are missing or empty in the DataFrame")
        # Return an empty figure if data is not correctly formatted
        return go.Figure()

@app.callback(
    Output('election-prediction', 'figure'),
    Input('election-prediction', 'id')
)
def update_election_prediction(_):
    outcome = df['predicted_label'].value_counts(normalize=True) * 100
    fig = go.Figure(data=[go.Pie(labels=outcome.index, values=outcome.values, hole=.4)])
    return fig

@app.callback(
    Output('lime-explanation-graph', 'figure'),
    Input('lime-explanation-graph', 'id')
)
def update_lime_explanation(_):
    idx = 0  # Index of the instance to explain
    exp = explainer.explain_instance(X_counts[idx].toarray()[0], model.predict_proba, num_features=10, top_labels=1)
    # Extract feature names and weights from the explanation
    features = [f for f, _ in exp.as_list()]
    weights = [w for _, w in exp.as_list()]
    # Create a horizontal bar chart
    fig = go.Figure(data=[go.Bar(
        y=features,
        x=weights,
        orientation='h'
    )])
    # Update layout for better readability
    fig.update_layout(
        title="LIME Explanation",
        xaxis_title="Weight",
        yaxis_title="Feature",
        yaxis={'categoryorder': 'total ascending'},  # Sort features by total value
        margin=dict(l=150)  # Increase left margin for long feature names
    )
    return fig


@app.callback(
    Output('k-anonymity-impact', 'figure'),
    Input('k-anonymity-impact', 'id')
)
def update_k_anonymity_impact(_):
    data = {'Before K-Anonymity': len(df), 'After K-Anonymity': len(df_k_anon)}
    fig = px.bar(x=list(data.keys()), y=list(data.values()), title="Impact of K-Anonymity on Data")
    # Adjusting bar width
    fig.update_traces(width=0.2)  # Adjust bar width as needed
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
