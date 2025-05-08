'''
Word embedding with Global Vectors (GloVe):
    Word-word co-occurrences within context windows may carry rich semantic information. For example, in a large corpus word “solid” 
    is more likely to co-occur with “ice” than “steam”, but word “gas” probably co-occurs with “steam” more frequently than “ice”. 
    Besides, global corpus statistics of such co-occurrences can be precomputed: this can lead to more efficient training.

The skip-gram model can be interpreted using global corpus statistics such as word-word co-occurrence counts.
The cross-entropy loss may not be a good choice for measuring the difference of two probability distributions, 
especially for a large corpus. GloVe uses squared loss to fit precomputed global corpus statistics.
The center word vector and the context word vector are mathematically equivalent for any word in GloVe.
GloVe can be interpreted from the ratio of word-word co-occurrence probabilities.

'''


'''
Subword Embedding
> The fastText Model
> Byte Pair Encoding
'''