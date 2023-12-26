import numpy as np 
from keras.models import Sequential # The base keras model class
from keras.layers import Dense, Activation # Dense is a fully connected layer of neurons 
from keras.optimizers import SGD # Stochastic gradient descent, but there are other 

# Out examples for an exclusive OR
x_train = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]]) # x_train is a list of samples of 2d feature vectors used for training
y_train = np.array([[0],[1],[1],[1]]) # y_train is desired outcomes (target values) for each feature vector sample

# Implementing keras
model = Sequential()
num_neurons = 10 
model.add(Dense(num_neurons, input_dim = 2)) # input_dim is only necessary for first layerl subsequent layers will calculate the shape automatically from output_dim
# we have 2d feature vectors for out 2-input XOR gate examples
model.add(Activation('tanh'))
model.add(Dense(1)) # output layer has one neuron to output a single binary classification value (0, 1)
model.add(Activation('sigmoid'))
model.summary()

# Training model
sgd = SGD(learning_rate=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

