# Keras network with one convolution layer
from keras.models import Sequential
from keras.layers import Conv1D

model = Sequential() 
model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', strides=1, input_shape=(100, 300)))

# Import keras convolution tools
import numpy as np 
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import GlobalMaxPool1D

# Preprocessor to load documents
import glob
import os

from random import shuffle

def pre_process_data(filepath):
    """
    This is dependent on your training data source but we will
    try to generalize it as best as possible.
    """
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

dataset = pre_process_data('Day 21/content/aclImdb_v1/aclImdb/train')
dataset[0]

# Vectorizer and tokenizer

from nltk.tokenize import TreebankWordDetokenizer
from gensim.models.keyedvectors import KeyedVectors
from nlpia.loaders import get_data
word_vectors = get_data('w2v', limit=200000)

def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordDetokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass # No matching token in google w2v vocab
        vectorized_data.append(sample_vecs)
    return vectorized_data

# Target labels
def collect_expected(dataset):
    '''
    Peel off the target values from the dataset
    '''
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected

vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)


# Train / Test Split
split_point = int(len(vectorized_data)*.8)

x_train = vectorized_data[split_point:]
y_train = vectorized_data[:split_point]
x_test = vectorized_data[split_point:]
y_test = vectorized_data[:split_point]

# CNN parameters
maxlen = 400
batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

# Padding and truncating token sequences

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

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)


model = Sequential()
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu',strides=1, input_shape=(maxlen, embedding_dims)))
model.add(GlobalMaxPool1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs = epochs, validation_data=(x_test, y_test))


sample_1 = "I hate that the dismal weather had me down for so long, when will it break! Ugh, when does happiness return? The sun is blinding and the puffy clouds are too thin. I can't wait for the weekend."

# Prediction
vec_list = tokenize_and_vectorize([(1, sample_1)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
model.predict(test_vec)