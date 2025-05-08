maxlen = 400
batch_size = 32
embedding_dims = 300
epochs = 2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
num_neurons = 50
model = Sequential()
model.add(LSTM(num_neurons, return_sequences=True,
               input_shape=(maxlen, embedding_dims)))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile('rmsprop','binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# Load and prepare the IMDB data
import numpy as np

# Similar to earlier models
# Data preprocessor
def pre_process_data(filepath):
    '''
    Load pos and neg examples from separate dirs then shuffle them together.
    '''
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')
    pos_label = 1
    neg_label = 0
    dataset = []
    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as f:
            dataset.append((pos_label, f.read()))
    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as f:
            dataset.append((neg_label, f.read()))
    shuffle(dataset)
    return dataset


# Data tokenizer + vectorizer
def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordDetokenizer()
    vectorized_data = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass
        vectorized_data.append(sample_vecs)
    return vectorized_data


# Target unzipper
def collect_expected(dataset):
    '''
    Peel off the target values from the dataset
    '''
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected
def pad_trunc(data, maxlen):
    '''
    For a given dataset pad with zero vectors or truncate to maxlen
    '''
    new_data = []
    # create a vector of 0s the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            # Append the appropriate number 0 vectors to the list
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data

import glob
import os
from random import shuffle
from nltk.tokenize import TreebankWordDetokenizer
from nlpia.loaders import get_data 
word_vectors = get_data('wv')
dataset = pre_process_data('./../aclimdb/train')
vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)
split_point = int(len(vectorized_data)*.8)

# used 1000 due to less memory
x_train = vectorized_data[:1000]
y_train = expected[:1000]
x_test = vectorized_data[19800:]
y_test = expected[19800:]

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

# Fit lstm model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))




sample_1 = """I hate that the dismal weather had me down for so long,
when will it break! Ugh, when does happiness return? The sun is
blinding and the puffy clouds are too thin. I can't wait for the
weekend."""
vec_list = tokenize_and_vectorize([(1, sample_1)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list,
(len(test_vec_list), maxlen, embedding_dims))

print(f"Sample's sentiment, 1 - pos, 2 - neg: {np.argmax(model.predict(test_vec),axis=1)}")
