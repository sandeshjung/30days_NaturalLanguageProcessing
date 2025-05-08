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

# Assemble a character-based LSTM model for generating text
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()


epochs = 6
batch_size = 128
model_structure = model.to_json()
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("best_weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True)
with open("shakes_lstm_model.json", "w") as json_file:
    json_file.write(model_structure)
model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint], validation_split=0.2)

# Sampler to generate character sequence
import random
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate threetexts with three diversity levels
import sys
start_index = random.randint(0, len(text) - maxlen - 1)
for diversity in [0.2, 0.5, 1.0]:
    print()
    print('----- diversity:', diversity)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('---- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence): #seed the trained network and see what it spits out as next character
            x[0, t, char_indices[char]] = 1
        preds = model.predict(x, verbose=0)[0] #model makes prediction
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index] #look up which character that index represents
        generated += next_char
        sentnece = sentence[1:] + next_char # add it to the seed and drop the first character to keep the length the same
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
    