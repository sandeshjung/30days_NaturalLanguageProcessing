# Preparing the corpus for training
'''
The training data will detemine the set of characters the encoder and decoder will support during
the training and during generation phase.
'''
# Build character sequence-to-sequence training set

from nlpia.loaders import get_data
df = get_data('moviedialog')
input_texts, target_texts = [], [] # the arrays hold input and target text read from corpus
input_vocabulary = set() # hold the seen characters in input and target text
output_vocabulary = set()
start_token = '\t'
# the target sequence is annotated with a start(first) and stop(last) token; the character representing the tokens
# are defined here. 
stop_token = '\n'
max_training_samples = min(25000, len(df) - 1) # defines how many lines are used for training

for input_text, target_text in zip(df.statement, df.reply):
    target_text = start_token + target_text + stop_token # target text needs to be wrapped with start and stop tokens
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_vocabulary:
            input_vocabulary.add(char)
    for char in target_text:
        if char not in output_vocabulary:
            output_vocabulary.add(char)

# Building character dictionary
# character sequence-to-sequence model parameters
input_vocabulary = sorted(input_vocabulary)
output_vocabulary = sorted(output_vocabulary)

input_vocab_size = len(input_vocabulary)
output_vocab_size = len(output_vocabulary)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_text])
input_token_index = dict([(char, i) for i, char in enumerate(input_vocabulary)])
target_token_index = dict([(char, i) for i, char in enumerate(output_vocabulary)])
reverse_input_char_index = dict((i,char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i,char) for char, i in target_token_index.items())


# Generate one-hot encoded tensors training sets
# loop over each input and target sample, and over each character of each sample, and one-hot encode each character
import numpy as np

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, input_vocab_size), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, output_vocab_size), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, output_vocab_size), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text): #loop over each character of each sample
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text): # for traiing data for the decoder, create the decoder_input_data and decoder_target_data
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t>0:
            decoder_target_data[i, t-1, target_token_index[char]]=1



# Construct and train character sequence encoder-decoder network
from keras.models import Model
from keras.layers import Input, LSTM, Dense
batch_size = 64
epochs = 100
num_neurons = 256
encoder_inputs = Input(shape=(None, input_vocab_size))
encoder = LSTM(num_neurons, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None, output_vocab_size))
decoder_lstm = LSTM(num_neurons, return_sequences=True,
return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
initial_state=encoder_states)
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
metrics=['acc'])
model.fit([encoder_input_data, decoder_input_data],
decoder_target_data, batch_size=batch_size, epochs=epochs,
validation_split=0.1)