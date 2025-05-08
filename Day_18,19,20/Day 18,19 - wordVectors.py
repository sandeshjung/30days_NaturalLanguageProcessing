'''
Word2vec learns the meaning of words merely by processing a large corpus of unlabeled text. 
Models that learn by trying to predict the input using a lower-dimensional internal representation are called autoencoders.
If any word in your corpus has some quality, like placeness, peopleness, conceptness or feamaleness, all the other words will also be given
a score for these qualities in word vectors.
In LSA, words only had to occur in the same document to have their meaning rub off on each other and get incorporated into their wordtopic
vectors. For word2vec word vectors, the words must occur near each other- typically fewer than five words apart and within the same sentence. 
And, word2vec word vector topic weights can be added and subtracted to create new word vectors that mean something.

The Word2vec model contains information about the relationships between words, including similarity.

The Word2vec model “knows” that the terms Portland and Portland Timbers are roughly the same distance apart as Seattle and Seattle Sounders. And
those distances (differences between the pairs of vectors) are in roughly the same direction. So the Word2vec model can be used to answer the analogy.

After adding and subtracting word vectors, resultant vector will almost never exactly equal one of the vectors in 
word vector vocabulary.
'''

'''
Word vectors represent semantic meaning of words as vectors in the context of the training corpus. This allows not only to answer
analogy questions but also reason the meaning of words in more general ways with vector algebra. 
There are two possible ways to train Word2vec embeddings:
    > The skip-gram approach predicts the context of words (ooutput words) from a word of interest(input word)
    > The continous bag-of-words (CBOW) approach predicts the target word (output word) from the nearby words (input words).

'''

'''
Skip-Gram vs CBOW: When to use which approach
Skip-gram approach works well with small corpora and rare terms. With skip-gram approach, you'll have more examples due to the network structure.
But the continuous bag-of-words approach shows higher accuracies for frequent words and is much faster to train.
'''