'''
From the book: Natural Language Processing with Python (https://www.nltk.org/book/)
1. Web and Chat Text
2. Brown Corpus (convenient resurce for studying systematic differences between genres)
3. Reuters Corpus
4. Inaugural Address Corpus
5. Annotated Text Corpora
6. Corpora in Other Languages
7. Text Corpus Structure
8. Loading your own corpus
9. Conditional Frequency Distributions
10. Generating Random Text with Bigrams
11. Stopwords (high-frequency words like the, to and also that we sometimes want to filter out 
of a document before further processing.)
12. Comparative Wordlists
'''

import nltk
from nltk.corpus import brown
romance_texts = brown.words(categories='romance')
fdist = nltk.FreqDist(w.lower() for w in romance_texts)
modals = ['what','when','where','who','why']
for m in modals:
    print(m + ':', fdist[m], end=' ')


# --------------------------------------------------------------------------------------

genre_word = [(genre, word) 
              for genre in ['news','romance']
              for word in brown.words(categories=genre)]
len(genre_word)
genre_word[:4]                              # start genre
genre_word[-4:]                             # end genre

cfd = nltk.ConditionalFreqDist(genre_word)
cfd
cfd.conditions()

# --------------------------------------------------------------------------------------

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
cfd.tabulate(conditions=['news','romance'],
             samples=days)

cfd.plot(conditions=['news','romance'],
         samples=days
)

# --------------------------------------------------------------------------------------

sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']
list(nltk.bigrams(sent))

def generate_model(cfdist, word, num=15):
    '''
    Generate text based on a conditional frequency distribution.

    Parameters:
    - cfdist (nltk.probability.ConditionalFreqDist): Conditional frequency distribution.
    - word (str): Initial context word.
    - num (int): Number of words to generate (default is 15).

    In the loop, we print the current value of the variable 'word' and reset 'word' to be the most
    likely token in that context (using max()). The next time through the loop, we use that word
    as our new context.
    '''
    for i in range(num):
        print(word, end= ' ')
        word = cfdist[word].max()

nltk.download('genesis')
text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
cfd['living']
generate_model(cfd, 'living')


# ------------------------------------------------------------------------------------------------


# from nltk.corpus import stopwords
# stopwords.words('english')

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [ w for w in text if w.lower() not in stopwords ]
    return len(content) / len(text)

nltk.download('reuters')
content_fraction(nltk.corpus.reuters.words())




# ------------------------------------------------------------------------------------------------

nltk.download('swadesh')
from nltk.corpus import swadesh
swadesh.fileids()
swadesh.words('en')

fr2en = swadesh.entries(['fr','en'])
fr2en
translate = dict(fr2en)
translate['chien']



# ------------------------------------------------------------------------------------------------


'''
In this day, I learned about corpus in Python and nltk.corpus
'''