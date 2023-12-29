# Initializing the Word2vec Pretrained model:
from gensim.models.keyedvectors import KeyedVectors
# https://s3.amazonaws.com/d14j-distribution/GoogleNews-vectors-negative300.bin.gz
# wget -c "https://s3.amazonaws.com/d14j-distribution/GoogleNews-vectors-negative300.bin.gz"
# https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download&resourcekey=0-wjGZdNAUop6WykTtMip30g
word_vectors = KeyedVectors.load_word2vec_format("Day 18, 19, 20\content\GoogleNews-vectors-negative300.bin.gz", binary=True, limit=200000)

# Basic implementation
word_vectors.most_similar(positive=["read","books"], topn=5) # Similar words 
word_vectors.doesnt_match("potatoes rice milk cake laptop".split()) # Outputs the Laptop
word_vectors.similarity("princess", "queen") # Calculates the Cosine similarity

# Train your domain-specific word2vec model
from gensim.models.word2vec import Word2Vec

# Parameters to control word2vec model training
num_features, min_word_count, num_workers, window_size, subsampling = 300, 3, 2, 6, 1e-3

tokens = [
    ["to", "provide", "early", "intervention/early", "childhood", "special", "education", "services", "to", "eligible", "children", "and", "their", "families"],
    ["essential", "job", "functions"],
    ["participate", "as", "a", "transdisciplinary", "team", "member", "to", "complete", "educational", "assessments", "for"],
]

# Instantiating a word2vec model
model = Word2Vec(
    tokens, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=window_size, sample=subsampling
)

# word_vectors['phone']

# Distance between Illinois and Illini
import numpy as np 
import os 
from nlpia.loaders import get_data
from gensim.models.word2vec import KeyedVectors
wv = get_data('word2vec')
np.linalg.norm(wv['Illinois'] - wv['Illini'])   # Euclidean distance
cos_similarity = np.dot(wv['Illinois'], wv['Illini']) / (
    np.linalg.norm(wv['Illinois']) *\
    np.linalg.norm(wv['Illini'])
) 
cos_similarity
1 - cos_similarity  # Cosine distance


# Bubble Chart of US cities
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# us_300D = get_data('cities_us_wordvectors')
# us_2D = pca.fit_transform(us_300D.iloc[:, :300]) 

# import seaborn
# from matplotlib import pyplot as plt
# from chart_studio import plotly
# df = get_data('cities_us_wordvectors_pca2_meta')
# html = plotly(
#     df.sort_values('population', ascending=False)[:350].copy()\
#     .sort_values('population'),
#     filename='plotly_scatter_bubble.html',
#     x='x', y='y',
#     size_col='population', text_col='name', category_col='timezone',
#     xscale=None, yscale=None, # 'log' or None
#     layout={}, marker={'sizeref': 3000})


# Train your own document and word vectors
import multiprocessing
num_cores = multiprocessing.cpu_count() # gensim uses python's multiprocessing module to parellelize training on multiple CPU cores
from gensim.models.doc2vec import TaggedDocument, Doc2Vec  # gensim doc2vec model contsins word vector embeddings as well as document vectors for each document
from gensim.utils import simple_preprocess #it is a crude tokenizer that will ignore one-letter words and all punctuation
corpus = ['This is the first document ..', \
          'another document ...']

training_corpus = []
for i, text in enumerate(corpus):
    tagged_doc = TaggedDocument(\
        simple_preprocess(text), [i])
    training_corpus.append(tagged_doc)

model = Doc2Vec(vector_size=100, min_count=2, workers = num_cores, epochs=10) 
model.build_vocab(training_corpus)
model.train(training_corpus, total_examples=model.corpus_count, epochs = model.epochs)

model.infer_vector(simple_preprocess('This is a completely unseen document'), epochs=10) 
