# Import all the things
import glob
import os
from random import shuffle
from nltk.tokenize import TreebankWordDetokenizer
from nlpia.loaders import get_data 
word_vectors = get_data('wv')

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

# Load and prepare data
dataset = pre_process_data('./../aclimdb/train')
vectorized_data = tokenize_and_vectorize(dataset)
# vectorized_data[0]
expected = collect_expected(dataset)
split_point = int(len(vectorized_data) *.8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]


# Initialize network parameters
maxlen  = 400
batch_size = 32
embedding_dims = 300
epochs = 2

# load test and training data
import numpy as np

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

# Initialize empty keras network
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN
num_neurons = 50
model = Sequential()

# Add a recurrent layer
model.add(SimpleRNN(
    num_neurons, return_sequences=True,
    input_shape=(maxlen, embedding_dims))
)

model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
'''
To prevent overfitting u add a dropout layer to zero out 20% of those inputs, randomly chosen on each input example.
And then add a classifier. In this case, you have one class: 'Yes - positive sentiment - 1' or 'No - negative sentiment'
so you chose a layer with one neuron (Dense(1)) and sigmoid activation function. But dense layer expects a flat vector of n 
elements as input. And the data coming out of the SimpleRNN is a tensor 400 elements long, and each of those are 50 elements long.
You use the convenience layer, Flatten() , that keras provides to flatten the input from a 400x50 tensor to a vector 
20,000 elements long. Flatten layer is a mapping. That means the error is backpropagated from the last layer back to the appropriate output
in the RNN layer and each of those backpropagated errors are then backpropagated through time from the appropriate point in the 
output.
'''

# Compile rnn
model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test))

