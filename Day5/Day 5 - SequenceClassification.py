'''
In case of POS tagging, a variety of different sequence classifier models can be used to jointly
choose POS tags for all the words in a given sentence.
One sequence classification strategy, known as consecutive classification or greedy sequence classification, is to find the most 
likely class label for the first input, then to use that answer to help find the best label for the next input. The process can 
then be repeated until all of the inputs have been labeled.
'''
import nltk
from nltk.corpus import brown
def pos_features(sentence, i, history):
    '''
    This function extracts features for a given word in a sentence for part-of-speech tagging.
    
    Parameters:
        - sentence: List of words representing the input sentence.
        - i: Index of the current word in the sentence.
        - history: List of predicted tags for the sentence so far.
    
    Returns:
        A dictionary containing features for the specified word.
    '''
    features= {"suffix(1)": sentence[i][-1:],
               "suffix(2)": sentence[i][-2:],
               "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
        features["prev-tag"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
        features["prev-tag"] = history[i-1]
    return features

class ConsecutivePosTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = pos_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = pos_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)
    
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
tagger = ConsecutivePosTagger(train_sents)
print(tagger.evaluate(test_sents))


'''
One shortcoming of this approach is that we commit to every decision that we make. For example, 
if we decide to label a word as a noun, but later find evidence that it should have been a verb, there's no way 
to go back and fix our mistake. One solution to this problem is to adopt a transformational strategy instead. 
Transformational joint classifiers work by creating an initial assignment of labels for the inputs, 
and then iteratively refining that assignment in an attempt to repair inconsistencies between related inputs.

Another solution is to assign scores to all of the possible sequences of part-of-speech tags, and to choose the sequence 
whose overall score is highest. This is the approach taken by Hidden Markov Models. Hidden Markov Models are similar to 
consecutive classifiers in that they look at both the inputs and the history of predicted tags. However, rather than simply 
finding the single best tag for a given word, they generate a probability distribution over tags. These probabilities are then 
combined to calculate probability scores for tag sequences, and the tag sequence with the highest probability is chosen.

'''

# Sentence Segmentation
sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for sent in sents:
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset-1)

def punct_features(tokens, i):
    return {'next-word-capitalized': tokens[i+1][0].isupper(),
            'prev-word': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1}

featuresets = [(punct_features(tokens,i), (i in boundaries))
               for i in range(1, len(tokens)-1)
               if tokens[i] in '.?!']

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)