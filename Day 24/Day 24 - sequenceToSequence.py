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