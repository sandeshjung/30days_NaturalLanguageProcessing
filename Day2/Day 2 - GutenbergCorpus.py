'''
A corpus (pluroal: corpora) refers to a collection of text documents or a body of written
or spoken text that is used for linguistic analysis, research, or language model training.
'''

import nltk
nltk.download('gutenberg')
nltk.corpus.gutenberg.fileids()                 # Gutenberg corpus

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)                                       # No of words it contain

emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance('surprize')

from nltk.corpus import gutenberg
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars/num_words),round(num_words/num_sents), round(num_words/num_vocab), fileid)

'''
Displays three statistics for each text:
average word length, average sentence length, and the number of times each vocabulary item appears 
in the text on average.
'''

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
macbeth_sentences
macbeth_sentences[1116]
longest_len = max(len(s) for s in macbeth_sentences)
[ s for s in macbeth_sentences if len(s) == longest_len]


'''
In this tutorial, I basically learned how to use access methods in a corpus.
'''