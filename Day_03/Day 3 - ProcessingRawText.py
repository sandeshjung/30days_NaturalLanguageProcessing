'''
Processing Raw text
'''

import nltk, re, pprint
from nltk import word_tokenize
from urllib import request


# Accessing the English translation of crime and punishments
url = 'http://www.gutenberg.org/files/2554/2554-0.txt'
response = request.urlopen(url)
raw = response.read().decode('utf-8')
# type(raw)
# len(raw)
# raw[:75]

tokens = word_tokenize(raw)
# type(tokens)
# len(tokens)
# tokens[:10]

text = nltk.Text(tokens)
# type(text)
# text[1024:1062]
text.collocations()


# Accessing the HTML documents
url = 'http://news.bbc.co.uk/2/hi/health/2284783.stm'
html = request.urlopen(url).read().decode('utf-8')
# html[:60]
from bs4 import BeautifulSoup
raw = BeautifulSoup(html, 'html.parser').get_text()
tokens = word_tokenize(raw)
tokens = tokens[110:390]
text = nltk.Text(tokens)
text.concordance('gene')



'''
In this tutorial, I completed processing the text from electronic books and from html documents.
Also,
Hierarchy in WordNet:

Synsets: WordNet organizes words into synsets, which are sets of synonyms representing a specific concept or meaning. 
Each synset corresponds to a unique concept, and words within the synset are considered synonymous.
Hypernyms: WordNet establishes a hierarchical structure by defining hypernym relationships. 
A hypernym is a more general term that encompasses a more specific term. For example, "vehicle" is a hypernym of "car."
Hyponyms: Conversely, hyponyms are more specific terms within a hierarchy. For instance, "rose" is a hyponym of "flower."
Semantic Similarity:

Path Similarity: One way to measure the semantic similarity between two synsets is by examining the shortest path that connects 
them in the hypernym hierarchy. The similarity is often calculated as the inverse of the path length, so that more specific 
terms have a higher similarity.
Leacock-Chodorow Similarity: This method considers the length of the shortest path between two synsets, taking into account 
the depth of the hierarchy. It adjusts the path length by considering the depth of the hierarchy, assuming that terms higher 
up in the hierarchy are more general.
Wu-Palmer Similarity: Similar to the Leacock-Chodorow method, Wu-Palmer similarity also considers the depth of the hierarchy. 
It calculates the similarity as 2 times the depth of the lowest common subsumer (ancestor), divided by the sum of the depths 
of the two synsets.
'''