'''
TF-IDF 'importance' scores worked not only for words, but also for short sequences of words, n-grams. These importance
scores for n-grams are great for searching text if you know the exact words or n-grams you're looking for.
'''
'''
LSA is an algorithm to analyze tf-idf matrix to gather up words into topics. It works on bag-of-words vectors, too, but tf-idf
vectors give slightly better results.
It also optimizes these topics to maintain diversity in the topic dimensions; when you use these new topics instead of the original words,
you still capture much of the meaning(semantics) of the document.
LSA is often referred to as a dimension reduction technique. LSA reduces the number of dimensions you need to 
capture the meaning of your document.
'''

'''
Two algorithms similar to LSA:
1) Linear discriminant analysis(LDA)
2) Latent Dirichlet allocation (LDiA)
'''
import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120
sms = get_data('sms-spam')

# creating dataframe
index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['spam'] = sms.spam.astype(int)
len(sms)
sms.spam.sum()
sms.head(6)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

# tokenization and tfidf
tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs.shape

# implementing LDA 
mask = sms.spam.astype(bool).values
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)
spam_centroid.round(2)

# calculating line between spam and not spam centroids
spamminess_score = tfidf_docs.dot(spam_centroid -ham_centroid)
spamminess_score.round(2)

# inspecting lda classifier result
from sklearn.preprocessing import MinMaxScaler
sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score > .5).astype(int)
sms['spam lda_predict lda_score'.split()].round(2).head(6)

# inspecting model evaluation
(1. - (sms.spam - sms.lda_predict).abs().sum() / len(sms)).round(3)
