'''
Encoder-decoder architecture
    The first half of an encoder-decoder model is the sequence encoder, a network which turns a sequence, such as natural
    language text, into a lower-dimensional representation.
    The other half is the sequence decoder. A sequence decoder can be designed to turn a vector back into human readable text again.

    @Whenever you train any neural network model, each of the internal layers contains all the information you need to solve the problem
    you trained it on. That information is usually represented by a fixed-dimensional tensor containing the weights or the activations of
    that layer. And if your network generalizes well, you can be sure that an information bottleneck exists-a layer where the number
    of dimensions is at a minimum. 

    For a single LSTM to work, you would need input and output sequences to have the same sequence lengths, and for translation they rarely do. 
    Sequence-to-sequence networks, sometimes abbreviated with seq2seq, solve this limitation by creating an input representation in the form
    of a thought vector. Sequence-to-sequence models then use that thought vector, sometimes called a context vector, as a starting point to
    a second network that recieves a different set of inputs to generate the ouput sequence.
            Word vectors are a compression of the meaning of a word into a fixed length vector. Words with similar meaning are close to each other
            in this vector space of word meanings. A thought vector is very similar. A neural network can compress imformation from any natural language
            statement, not just a single word, into a fixed length vector that represents the content of the input text. They are used as a numerical
            representation of the thought within a document to drive some decoder model, usually a translation decoder.
    A sequence-to-sequence network consists of two modular recurrent networks with a thought vector between them. The encoder outputs a thought vector at 
    the end of its input sequence. The decoder picks up that thought and outputs a sequence of tokens.
    The thought vector has two parts, each a vector: the output (activation) of the hidden layer of the encoder and the memory state of the LSTM cell
    for that input example.The thought vector then becomes the input to a second network: the decoder network.
    The second network uses that initial state and a special kind of input, a start token to learn to generate the first element of the target sequence.

    A variational autoencoder is a modified version of an autoencoder that is trained to be a good generator as well as encoder-decoder. A variational
    autoencoder produces a compact vector that not only is a faithful representation of the input but is also Gaussian distributed. This makes it easier to generate
    new output by randomly selecting a seed vector and feeding that into the decoder half of the autoencoder.

    
'''


'''
Sequence-to-sequence pipeline
1. Preparing dataset for the sequence-to-sequence training
    Prepare target data and pad it to match the longest target sequence. Also, the sequence lengths of the input and target data don't need to be the same.
    In addition to the required padding, the output sequence should be annotated with the start and stop tokens, to tell the decoder when the job starts and
    when it's done.
    Each training example for the sequence-to-sequence model will be a triplet: initial input, expected output(prepened by a start token), and 
    expected output(without the start token).
2. Sequence to Sequence model in keras
    During the training phase, youll train the encoder and decoder network together, end to end, which requires three data points for each sample: a training
    encoder input sequence, a decoder input sequence, and a decoder output sequence. 
    We need input and output sequence for the decoder is because we train the decoder with a method called teacher forcing, where we'll use the initial state
    provided by the encoder network and train the decoder to produce the expected sequences by showing the input to the decoder and letting it predict the
    same sequence.
3. Sequence encoder
    The encoder's sole prupose is the creation of your thought vector, which then serves as the initial state of the decoder netwrork. You can't train an encoder
    fully in isolation. The backpropagation that will train the encoder to create an appropriate thought vector will come from the error that's generated
    later downstream in the decoder. 
    # Thought encoder in keras
    encoder_inputs = Input(shape=(None, input_vocab_size))
    encoder = LSTM(num_neurons, return_state=True) # return_state=True to return internal states
    encoder_outputs, state_h, state_c = encoder(encoder_inputs) # first return value of the LSTM is the output of the layer
    encoder_states = (state_h, state_c)
4. Thought decoder
    Similar to encoder, major difference is that his time you do wnat to capture the output of the network at each time step. You want to judge the 'correctness'
    of the output, token by token. this is where you use the second and third pieces of the sample 3-tuple. The decoder has a standard token-by-token input and 
    a token-by-token output. 
    You want the decoder to learn to reproduce the tokens of a given input sequence given the state generated by the first piece of the 3-tuple fed into 
    the encoder.

    To calculate the error of the training step, you'll pass the output of your LSTM layer into a dense layer. The dense layer will have a number of neurons
    equal to the number of all possible output tokens. The dense layer will have a softmax activation function across those tokens. SO at each time step,
    the network will provide a probability distribution over all possible tokens for what is thinks is mose likely the next sequence element. 

    # Thought decoder in keras
    decoder_inputs = Input(shape=(None, output_vocab_size))
    decoder_lstm = LSTM(num_neurons, return_sequences=True, return_state=True)
    decoder_outputs , _, _ = decoder_lst,(decoder_inputs, initial_state=encoder_states) #functional api allwos to pass initial state to the LSTM layer 
    by assigning the last encoder state to initial_state
    decoder_dense = Dense(output_vocab_size, activation='softmax') 
    decoder_outputs = decoder_dense(decoder_outputs) # Passing output of LSTM layer to the softmax layer
5. Assembling the sequence-to-sequence network
    The functional API of Keras allows you to assemble a model as object calls. The Model object lets you define its input and output parts of the network. 
    For this sequence-tosequence network, you’ll pass a list of your inputs to the model.
    # Keras functional API (Model())
    model = Model(
        inputs = [encoder_inputs, decoder_inputs],
        outputs = decoder_outputs
    )
'''