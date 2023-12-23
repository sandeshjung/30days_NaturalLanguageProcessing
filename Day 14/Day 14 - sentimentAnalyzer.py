import nltk
import pandas as pd
# VADER: A rule based sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# pip install vaderSentiment
sa = SentimentIntensityAnalyzer()
sa.lexicon 
sa.polarity_scores('Python is good for Natural Language Processing')

# Naive Bayes: Data Preparation
from nlpia.data.loaders import get_data 
movies = get_data('hutto_movies')
movies.head().round(2)

# Implementation of Casual Tokenizer
from nltk.tokenize import casual_tokenize
from collections import Counter
bag_of_words = []
for text in movies.text:
    bag_of_words.append(Counter(casual_tokenize(text)))
df_bows = pd.DataFrame.from_records(bag_of_words)
df_bows = df_bows.fillna(0).astype(int)
df_bows.head()

