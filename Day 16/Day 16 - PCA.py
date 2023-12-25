'''
Principal component analysis is another name for SVD when it's used for dimension reduction.
sklearn.PCA automatically 'centers' your data by substracting off the mean word frequencies. Another, more subtle trick is that PCA uses a function called flip_sign to
deterministically compute the sign of the singular vectors. Finally, the sklearn implementation of PCA implements an optional 'whitening' step. 
'''
# PCA on 3d vectors
import pandas as pd
pd.set_option('display.max_columns', 6) # Ensure that your pd.DataFrame printouts fit within the width of a page
from sklearn.decomposition import PCA
import seaborn
import matplotlib.pyplot as plt 
from nlpia.data.loaders import get_data

df = get_data('pointcloud').sample(1000)
pca = PCA(n_components=2) # reducing a 3d point cloud to a 2d 'projection' for display in a 2d scatter plot
df2d = pd.DataFrame(pca.fit_transform(df), columns=list('xy'))
df2d.plot(kind='scatter', x='x', y='y')
plt.show()

# Stop horsing 
pd.options.display.width = 120      # helps the wide pandas dataframes print out a bit prettier
#@ Creating the dataframe
sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)] #adding an exclamation mark to the sms message index no to make them easier to spot
sms.index = index
sms.head(6)

# to calculate the TF-IDF vectors for each of these messages
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
len(tfidf.vocabulary_)

tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean() # this centers your vectorized documents (BOW vectors) by subtracting the mean
tfidf_docs.shape
sms.spam.sum()


#@ Using PCA for sms message semantic analysis
from sklearn.decomposition import PCA 

pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
pca_topic_vectors.round(3).head(6)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=16, n_iter=100) 
svd_topic_vectors = svd.fit_transform(tfidf_docs.values) # fit_transpose decomposes tf-idf vectors and transforms them into topic vectors in one step
svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns, index=index)
svd_topic_vectors.round(3).head(6)

'''
To find out how well a vector space model will work for classification is to see how cosine similarities between vectors correlate with membership in the same class.
'''

import numpy as np

svd_topic_vectors = (svd_topic_vectors.T / np.linalg.norm(svd_topic_vectors, axis=1)).T #Normalizing each topic vector by its length (L2-norm) allows you to compute the cosine distances with dot product
svd_topic_vectors.iloc[:10].dot(svd_topic_vectors.iloc[:10].T).round(1)


# LDiA works with raw BOW count vectors rather than normalized TF-IDF vectors
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
np.random.seed(42)

counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(), index=index)
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(), counter.vocabulary_.keys())))
# zip is useful if you'd like to iterate over multiple iterables simultaneously
bow_docs.columns = terms

from sklearn.decomposition import LatentDirichletAllocation as LDiA

ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(bow_docs) # LDiA takes a bit longer than PCA or SVD
ldia.components_.shape
# model has allocated 9232 words to 16 topics
pd.set_option('display.width', 75)
components = pd.DataFrame(ldia.components_.T, index=terms, columns=columns)
components.round(2).head(3)
# So the exclamation point term (!) was allocated to most of the topics, but is a particularly strong part of topic3 where the quote symbol (") is hardly playing a role at all. 
# Perhaps “topic3” might be about emotional intensity or emphasis and doesn’t care
# much about numbers or quotes.
components.topic3.sort_values(ascending=False)[:10]

ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, index=index, columns=columns)
ldia16_topic_vectors.round(2).head()

# LDiA + LDA = spam classifier
# using LDiA topic vectors to train an LDA model

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size=0.5, random_state= 271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
round(float(lda.score(X_test, y_test)), 2)


'''
Lets find out how LDiA model compares to a much higher-dimensional model based on the tfidf vectors. Since, tfidf has more features which can experience overfitting and poor
generalization. This is where generalization of LDiA and PCA should help.
'''

tfidf_docs_2 = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs_2 = tfidf_docs_2 - tfidf_docs_2.mean(axis=0)

X_train, X_test, y_train, y_test = train_test_split(tfidf_docs_2, sms.spam.values, test_size=0.5, random_state=271828)
lda = LDA(n_components=1) 
lda = lda.fit(X_train, y_train) # fitting an LDA model to all these thousands of features will take a quite long time.
round(float(lda.score(X_train, y_train)), 3) # overfitting
round(float(lda.score(X_test, y_test)), 3)
'''
The training set accuracy for your TF-IDF based model is perfect! But the test set accuracy is much worse than when you trained it on lower-dimensional topic vectors
instead of TF-IDF vectors.
'''