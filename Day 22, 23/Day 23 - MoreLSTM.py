from nltk.corpus import gutenberg

gutenberg.fileids()

# Preprocess Shakespeare plays
text = ''
for txt in gutenberg.fileids(): # Concatenate all Shakespeare plays in the Gutenberg corpus in NLTK
    if 'shakespeare' in txt:
        text += gutenberg.raw(txt).lower()
chars = sorted(list(set(text)))
char_indices = dict((c,i)   # make dict of characters to an index, for reference in one hot encoding
                    for i, c in enumerate(chars))
indices_char = dict((i,c)   # make opposite dict for lookup when interpreting the one-hot encoding back to character
                    for i,c in enumerate(chars))
'corpus length: {} total chars: {}'.format(len(text), len(chars)) 

print(text[:500])

# Assemble a training set

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i+ maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

# One-hot encode the training examples
import numpy as np
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1