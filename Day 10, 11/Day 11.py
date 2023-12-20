'''
Word Embedding (word2vec):
word vectors are vectors used to represent words, and can also be considered as feature vectors or representations of words. 
The technique of mapping words to real vectors is called word embedding
    1. One-Hot Vectors 
        To obtain one-hot vector representation for any word with index i, we create a length-N vector with all 0s and set the element
        at position i to 1.
        Bad choice because it cannot accurately express the similarity between different words, such as cosine similarity.
    2. Self-Supervised word2vec
        word2vec maps each word to a fixed-length vector, and these vectors can better express the similarity and analogy relationship
        among different words. 
        it contains two models, skip-gram and continuous bag of words (CBOW).
        For semantically meaningful representations, their training relies on conditional probabilities that can be viewed as predicting some 
        words using some of their surrounding words in corpora.

> The Skip-Gram model: assumes that the a word can be used to generate its surrounding words in a text sequence. 
    In training, we learn model parameters by maximizing the likelihood function.
> CBOW: similar to skip-gram. cbow assumes that a center word is generated based on its surrounding context words in the text sequence.
    
'''


# Pretraining word2vec
import math
import torch 
from torch import nn
from d2l import torch as d2l

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size, num_noise_words)

# Skip-gram Model
# Embedding Layer : maps a token's index to its feature vector
# The weigth of this layer is a matrix whose number of rows equals to dictionary size (input_dim) and num of columns equals to vector dimension
# for each token (output_dim)
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape},'
      f'dtype={embed.weight.dtype})')

# Since the vector dimension (output_dim) was set to 4, the embedding layer returns vectors with shape (2,3,4) for a minibatch of token indices with (2,3)

x = torch.tensor([[1,2,3], [4,5,6]])
embed(x)

# Defining the forward propagation

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

skip_gram(torch.ones((2,1), dtype=torch.long),
          torch.ones((2,4), dtype=torch.long), embed, embed).shape


#  Training 

# Binary Cross Entropy Loss
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction='none')
        return out.mean(dim=1)
    
loss = SigmoidBCELoss()

pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)

# Initializing Model Parameters
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))

# Defining the Training loop
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')       
            
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])