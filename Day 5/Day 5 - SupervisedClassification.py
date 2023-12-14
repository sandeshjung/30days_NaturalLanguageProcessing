'''
Classification is the task of choosing the correct class label for a given input.
The classification is said to be supervised classification if it is built based 
on training corpora containing the correct label for each input. 
eg. deciding whether an Email is spam or not.

'''

# Gender Identification
import nltk
from nltk.corpus import names
nltk.download('names')
labeled_names = ([(name, 'male') for name in names.words('male.txt')]) + [(name, 'female') for name in names.words('female.txt')]
import random
random.shuffle(labeled_names)
# print(labeled_names)

def gender_features(word):
    return {'last_letter': word[-1]}

# we use the feature extractor to process the names data, and divide the resulting list of feature sets into a training set and a test set
featuresets=[(gender_features(n), gender) for (n, gender) in labeled_names]
# print(featuresets)
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))


print(f'accuracy: {nltk.classify.accuracy(classifier, test_set)}')

classifier.show_most_informative_features(5)

# When working with large corpora, constructing a single list that contains the features of every instance can use up a large amount of memory
from nltk.classify import apply_features
train_set = apply_features(gender_features, labeled_names[500:])
test_set = apply_features(gender_features, labeled_names[:500])

'''
feature extractors are built through a process of trial-and-error, guided by intuitions about what information is relevant to the problem. 
It's common to start with a "kitchen sink" approach, including all the features that you can think of, and then checking to see which 
features actually are helpful.
'''

def gender_features2(name):
    features = {}
    features['first_letter'] = name[0].lower()
    features['last_letter'] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count({})'.format(letter)] = name.lower().count(letter)
        features['has({})'.format(letter)] = (letter in name.lower())
    return features

gender_features2('John')

'''
However, there are usually limits to the number of features that you should use with a given learning algorithm — 
if you provide too many features, then the algorithm will have a higher chance of relying on idiosyncrasies of your 
training data that don't generalize well to new examples. This problem is known as overfitting
'''
