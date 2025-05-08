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

# Naive Bayes: Sentiment Analysis
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb = nb.fit(df_bows, movies.sentiment > 0)
# Processing the Classifiers
movies["predicted_sentiment"] = nb.predict_proba(df_bows)[:, 1]
movies["error"] = (movies.predicted_sentiment - movies.sentiment).abs()
movies.error.round(1)
movies["sentiment_ispositive"] = (movies.sentiment > 0).astype(int)
movies["predicted_ispositive"] = (movies.predicted_sentiment > 0).astype(int)


'''
In this tutorial, I have read and implemented about bag of words, vectorizing concept, vector spaces,
cosine similarity, Zipf's law and inverse frequency concept, text modeling, tfidf, relevance
ranking and okapi BM25 concept.
'''

# Bag of words

from nltk.tokenize import TreebankWordTokenizer
sentence = """The faster Harry got to the store, the faster Harry,
the faster, would get home."""
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
tokens
from collections import Counter
bag_of_words = Counter(tokens)
bag_of_words
bag_of_words.most_common(4)


from nlpia.data.loaders import kite_text
tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)
import nltk
nltk.download('stopwords', quiet=True)
stopwords = nltk.corpus.stopwords.words('english')
tokens = [x for x in tokens if x not in stopwords]
kite_counts = Counter(tokens)
kite_counts

# vectorizing
document_vector = []
doc_length = len(tokens)
for key, value in kite_counts.most_common():
    document_vector.append(value / doc_length)
document_vector


docs = ["The faster Harry got to the store, the faster and faster Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")

# let's look at lexicon for this corpus
doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
len(doc_tokens[0])
all_doc_tokens = sum(doc_tokens, [])
len(all_doc_tokens)
lexicon = sorted(set(all_doc_tokens))
len(lexicon)
lexicon

'''
the concept of vector spaces is then applied to nlp, where documents are represented as vectors in a 
high-dimensional space. The dimensionality of the vector space is determined by the number of distinct words
in the corpus. 
Cosine similarity can be used as a measure of similarity between vectors, explaining its calculation and interpretation.
Cosine similarity ranges from -1(opposite vectors) to 1(identical vectors), providing a measure of how much vectors point in 
the same direction.
'''

'''
Zipf's law states that given some corpus of natural language utterances, the frequency of any word is inversely
proportional to its rank in the frequency table.
Specifically, inverse proportionality refers to a situation where an item in a ranked list will appear with a frequency
tied explicitly to its rank in the list.
'''

'''
In the context of large colleciton of documents, the author discusses the concept of TF-IDF which is a way of 
assigning weights to words based on how often they appear in a document and how common they are 
across the entire collection of documents(corpus).
for a given term, t, in a given document, d, in a corpus, D:
tf(t,d) = count(t)/count(d)
idf(t,D) = log(number of documents / number of documents containing t)
tf-idf(t, d, D) = tf(t,d) * idf(t,D)
'''

'''
to enhance document vectors with TF-IDF scores to better capture the meaning or topic of each document.
The TF-IDF values replace simple word counts in the vectors. The process involves TF-IDF scores for each word in each
document, resulting in a K-dimensional vector representation for each document in the corpus.
'''

'''
The author of the book suggests using TF-IDF vectors for a basic search engine. A search query is treated as a document, and 
its TF-IDF based vector representation is obtained. The documents with the highest cosine similarities to the query vector are 
considered the most relevant search results.
'''

'''
There is an improved approach to ranking search results known as Okapi BM25. Unlike simple cosine similarity,
Okapi BM25 normalizes and smoothens the similarity score, considering factors such as duplicate terms in the query document and 
clipping term frequencies for the query vector at 1. Additionally, the dot product for cosine similarity is normalized by a nonlinear
function of the document length.
'''

