import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import plotly.express as px

# Download the VADER lexicon
nltk.download("vader_lexicon", quiet=True)

# Initialize Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Title
st.title("Automated Sentiment Analysis Dashboard")

# Function to classify sentiment
def get_sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file containing feedback", type=["csv"])
if uploaded_file:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Ensure the Feedback column exists
        if "Feedback" not in data.columns:
            st.error("Error: The uploaded file must have a 'Feedback' column.")
        else:
            # Handle missing or null values
            data["Feedback"] = data["Feedback"].fillna("")

            # Perform sentiment analysis
            sentiment_scores = []
            sentiment_labels = []

            for feedback in data["Feedback"]:
                # Calculate sentiment score
                score = sid.polarity_scores(str(feedback))["compound"]
                sentiment_scores.append(score)

                # Get sentiment label
                label = get_sentiment_label(score)
                sentiment_labels.append(label)

            # Add sentiment data to the DataFrame
            data["Sentiment"] = sentiment_scores
            data["Sentiment Label"] = sentiment_labels

            # Display results
            st.write("Analyzed Data", data)

            # Sentiment Summary
            sentiment_summary = data["Sentiment Label"].value_counts()
            st.bar_chart(sentiment_summary)

            # Pie Chart
            fig = px.pie(data, names="Sentiment Label", title="Sentiment Distribution")
            st.plotly_chart(fig)

            # Download Button
            st.download_button(
                label="Download Results as CSV",
                data=data.to_csv(index=False),
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
