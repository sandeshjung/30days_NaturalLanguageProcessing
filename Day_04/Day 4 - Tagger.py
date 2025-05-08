'''
The process of classifying words into their parts of speech and labeling them
accordingly is known as part-of-speech tagging, POS-tagging or tagging.
POS are also called word classes or lexical categories.
POS-tagger processes a sequence of words, and attaches a part of speech tag to each word.
'''
# Automatic Tagging
import nltk
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

# Default tagger
raw = 'Hi my name is Sandesh, and I am learning NLP!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)
default_tagger.evaluate(brown_tagged_sents)

# Regular expression tagger
patterns = [
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # simple past
    (r'.*es$', 'VBZ'),                 # 3rd singular present
    (r'.*ould$', 'MD'),                # modals
    (r'.*\'s$', 'NN$'),                # possessive nouns
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')                      # nouns (default)
]

regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)

# Lookup tagger
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)
sent = brown.sents(categories='news')[3]
baseline_tagger.tag(sent)

'''
Evaluation:
    we evaluate the performance of a tagger relative to the tags a human expert would assign. Since we don't
    usually have access to an expert and impartial human judge, we make do instead with gold standard test data.
    This is a corpus which has been manually annotated and which is accepted as a standard against which the 
    guesses of an automatic system are assessed. 
    The tagger is regarded as being correct if the tag it guesses for a given word is the same
    as the gold standard tag.
'''

'''
One way to address the trade-off between accuracy and coverage is to use the more accurate algorithms when we can,
 but to fall back on algorithms with wider coverage when necessary. For example, we could combine the results of a bigram tagger, 
 a unigram tagger, and a default tagger, as follows:

Try tagging the token with the bigram tagger.
If the bigram tagger is unable to find a tag for the token, try the unigram tagger.
If the unigram tagger is also unable to find a tag, use a default tagger.
'''

size = int(len(brown_tagged_sents) * 0.9)
size
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)


'''
Today I have learned about Automatic Tagging such as default tagger, bigram tagger and so on. I have also 
learned about combining taggers using backoff parameter.
'''
