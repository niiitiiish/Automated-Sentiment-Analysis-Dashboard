import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
st.write("#Real time sentiment Analysis")
nltk.download('vader_lexicon')
user_input = st.text_input("Please Rate Our Services >>: ")
sid = SentimentIntensityAnalyzer()
score = sid.polarity_scores(user_input)
if score["compound"] >= 0.05:
    st.success("Positive Sentiment 😊")
elif score["compound"] <= -0.05:
    st.error("Negative Sentiment 😟")
else:
    st.info("Neutral Sentiment 😐")