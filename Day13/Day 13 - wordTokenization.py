'''
tokenizing text into words and n-grams, dealing with nonstandard punctuation and emoticons, and compressing 
token vocabulary through stemming and lemmatization. The goal is to build a vector representation of statements

challenges of stemming, which involves grouping various inflections of a word into the same cluster or "bucket." 
It emphasizes the difficulty of developing algorithms that can correctly group inflected forms of words based solely on their 
spelling. For example, removing verb endings like "ing" from "ending" to obtain a stem called "end" that represents both words 
is a complex task. Similarly, stemming "running" to "run" requires removing not only the "ing" but also the extra "n," 
while ensuring that words like "sing" remain unaffected.

The text further highlights the challenge of discriminating between a pluralizing "s" at the end of words like "words" and 
a normal "s" at the end of words like "bus" and "lens." 

'''

'''
Tokenization in NLP: Tokenization involves breaking up text into smaller chunks or segments, such as words or punctuation. 
The text emphasizes the importance of segmentation and focuses on segmenting text into tokens, which are usually words.

Tokenizer in Computer Science: The text draws a parallel between tokenization in NLP and tokenization in computer language compilers. 
In the context of a computer language compiler, a tokenizer is often referred to as a scanner, lexer, or lexical analyzer.

Vocabulary and Parser: The vocabulary (set of all valid tokens) in computer language processing is called a lexicon, and 
the parser in a compiler is compared to an NLP pipeline. Tokens are mentioned as the end of the line for context-free grammars 
used in parsing computer languages.

Importance of Tokenization: Tokenization is highlighted as the first step in an NLP pipeline and can significantly impact the 
rest of the pipeline. It breaks unstructured data (natural language text) into discrete elements, allowing the creation of 
numerical data structures suitable for machine learning.

One-Hot Vectors: The text introduces the concept of one-hot vectors, which are binary vectors with only one non-zero (hot) element. 
Each word in a document is represented as a one-hot vector, and the vocabulary is used to create these vectors.

Example Tokenization in Python: The text provides a Python example using the split method to tokenize a sentence into words. 
It then demonstrates the creation of one-hot vectors for each word in the sentence.

Pandas DataFrame for Representation: The text suggests using Pandas DataFrames to represent one-hot vectors, 
making it easier to visualize and work with the data.

Challenges and Considerations: Challenges such as handling punctuation, ordering of tokens, and the sparsity of one-hot vectors 
are acknowledged. The need for more sophisticated techniques, such as bag-of-words models, is introduced.

Efficiency Considerations: The text discusses the efficiency of one-hot vectors, highlighting their sparsity and the potential
need for dimension reduction when dealing with large vocabularies.

Bag-of-Words Models: The concept of bag-of-words models is introduced as an alternative to one-hot vectors for representing documents. 
Bag-of-words models focus on word frequencies, ignoring word order.

Binary Bag-of-Words Vector: The text describes how to represent a document using a binary bag-of-words vector, 
indicating the presence or absence of each word in the document.

DataFrame for Multiple Sentences: The text extends the example to include multiple sentences in a Pandas DataFrame, 
showcasing how the DataFrame can be used as a corpus representation.

'''
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer

sentence = """"Hello everyone, I am Sandesh Jung Kunwar \
and I am learning Natural Language Processing.""" 

token_sequence = str.split(sentence) #str.split() quick and dirty tokenizer
vocab = sorted(set(token_sequence)) # Unique tokens in the sentence
# ', '.join(vocab)
num_tokens = len(token_sequence) # Number of tokens
vocab_size = len(vocab) # Number of Unique tokens
# num_tokens, vocab_size
onehot_vectors = np.zeros((num_tokens, vocab_size), int) 
for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1
# ' '.join(vocab)
    # onehot_vectors

'''
A DataFrame keeps track of labels for each column, allowing you to label each column in our table with the 
token or word it represents. A DataFrame can also keep
track of labels for each row in the DataFrame.index, for speedy lookup.
'''
# pd.DataFrame(onehot_vectors, columns=vocab)

# df = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence.split()])),columns=['sent']).T
# df
corpus = {}
for i, sent in enumerate(sentence.split('\n')):
    corpus["sent{}".format(i)] = dict((tok, 1) for tok in sent.split())
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
pattern = re.compile(r"([-\s.,?!:;])+")
tokens = pattern.split(sentence)
tokens
list(filter(lambda x:x if x and x not in "- \t\n.,?!" else None, tokens)) #Implementation of lambda

#@ Implementation of Regexp Tokenizer
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
tokenizer.tokenize(sentence)

#@ Implementation of Treebank Tokenizer
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(sentence)

#@ Implementation of Casual TOkenizer
from nltk.tokenize.casual import casual_tokenize
casual_tokenize(sentence)

#@ Implementation of NGram Tokenizer
from nltk.util import ngrams
list(ngrams(tokens, 3))
