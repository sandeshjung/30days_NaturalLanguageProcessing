'''
Natural language processing:
    Natural Language processing is a field of Linguistics, Computer Science, and Artificial Intelligence concerned with 
    the interections between Computers and Human Language, in particular how to program computers to process
    and analyze large amounts of Natural Language Data. 
'''

# Fundamentals of Natural Language Processing

#Downlading all the necessary libraries and dependencies
import nltk
import re
import string

from nltk.corpus import stopwords       # Module for stop words (words used to eliminate words that are 
                                        #    so widely used that carry very little useful information)
from nltk.stem import PorterStemmer     # Module for stemming
from nltk.tokenize import TweetTokenizer # Module for tokenizing the strings

# Example of the tweets
tweet = "That book was amazing, as the contents contained in the book blows my mind. :) #amazing #great #curiousity https:///sandesh.c...g"

# Removing the Hyperlinks, Styles and Marks
print(f'Original tweet: {tweet}')
print(" ")
def remove_t(tweets):
    # Removing the retweet text
    tweet1 = re.sub(r'^RT[\s]+', '', tweets)
    # Removing the hyperlinks
    tweet1 = re.sub(r'https?:\/\/.*[\r\n]*','',tweet1)
    # Removing the hashtags
    tweet1 = re.sub(r'#', '', tweet1)
    return tweet1

tweet1 = remove_t(tweet)
print(f'New tweet: {tweet1}')


'''
In this tutorial, I learned about regular expressions (re) and how to remove punctuations and hyperlinks ffrom the the text provided.
'''


# ---------------------------------------------------------------------------------------------------------------------------

'''
String tokenization:
    In NLP, string tokenization is a process where the string is splitted into individual words or individual parts without blanks and tabs.
    In this step, the words in the String lowercased. 
    The Tokenize Module from NLTK makes very easy to carry out this process.
    Normally Tokenization is of two types:
        1. Word tokenzation: Dividing text into words. For example, the sentence "Tokenization is important!" would be 
        tokenized into ["Tokenization", "is", "important", "!"].
        2. Sentence tokenization: Diving text into sentences. This is useful for tasks that operate at the sentence level.
'''


# Instantiate the tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
'''
            preserve_case = False means that the tokenizer will convert the text to lower case
            strip_handles = True, the tokenizer will replace sequenes of repeated characters with only two occurrences.
            reduce_len = True, the tokenizer will replace sequences of repeated characters with only two occurrences.
'''

# Tokenizing the tweetrs
tweet_tokens = tokenizer.tokenize(tweet1)

# Importing the list of English stopwords from NTLK
eng_stopwords = stopwords.words('english')

# Removing the stopwords and punctuations from the tokenized tweets
clean_tweets = []
for word in tweet_tokens:
    if (word not in eng_stopwords and               # Removing the stopwords
        word not in string.punctuation):            # Removing the punctuations
        clean_tweets.append(word)

print(f'Cleaned and tokenized tweets:\n {clean_tweets}')        # Inspecting the cleaned and tokenized tweets
print(f'\nPunctuations: {string.punctuation}')                  # Inspecting the Punctuations


'''
In this tutorial, I learnt about String Tokenization, stop word and punctutation in NPL. I implemented TweetTokenizer and removed 
stopwords and punctuation from the tokenized tweets.
'''


# -----------------------------------------------------------------------------------------------------


'''
Stemming in NLP:
    Stemming is a process of converting a word to its most General form or Stem. It's basically the process of removing the suffix
    from a word and reduce it to it's root word. It helps in reducing the size of Vocabulary. 
    Types of Stemmers:
    1) Porter Stemmer: It is one of the most common and gentle stemmer which is very fast but not very precise.
    2) Snowball Stemmer: It's actual name is English Stemmer is more precise over large Dataset.
    3) Lancaster Stemmer: It is very aggrssive algorithm. It will hugely trim down the working data which itself has pros and cons.

    Reasons for stemming:
    > Text Normalization (eg. running, runs, and ran would be stemmmed to base form 'run')
    > Reducing Dimensionality 
    > Improving text retrieval 
    > Reducing spelling variations
'''

# Downloading the libraries and Dependencies
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

# Porter Stemmer
stemmer = PorterStemmer()                                   # Instantiating Stemmer class
stemWords = [stemmer.stem(word) for word in clean_tweets]
print(f"Porter Stemmer: {' '.join(stemWords)}")             # Inspecting the result

# Snowball Stemmer
stemmer = SnowballStemmer()                                 # Instantiating Snowball Stemmer class
stemWords = [stemmer.stem(word) for word in clean_tweets]
print(f"\nSnowball Stemmer: {' '.join(stemWords)}")         # Inspecting the result

# Lancaster Stemmer
stemmer = LancasterStemmer()                                # Instantiating Lancaster Stemmer class
stemWords = [stemmer.stem(word) for word in clean_tweets]
print(f"\nLancaster Stemmer: {' '.join(stemWords)}")        # Inspecting the result


'''
In this tutorial, I implemented porter, snowball and lancaster stemmers.
'''

# -----------------------------------------------------------------------------------------------------

'''
Lemmatization in NLP:
    Lemmatization is the process of grouping together the inflected forms of words so that they can be 
    analysed as a single item, identified by the word's Lemma or a Dictionary form. It is the process where 
    individual tokens from a sentence or words are reduced to their base form. 
    Lemmatization is much more informative than simple stemming. 
    eg.
        Original tokens: ['The', 'cats', 'are', 'running', 'and', 'the', 'mice', 'are', 'hiding', '.']
        Lemmatized tokens: ['The', 'cat', 'are', 'running', 'and', 'the', 'mouse', 'are', 'hiding', '.']

'''

# Downloading the libraries and dependencies
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from spacy.lookups import Lookups

nlp = spacy.load('en_core_web_sm')
def Lemmatizer(token):                              # In Spacy v3, spacy.lemmatizer doesnt work
    return token.lemma_                             # default lemmatizer

words_l = ['dogs','cats','apples','sings','brings'] # Example of words to be lemmatized

# In SpaCy v2
# Implementation of Lemmatization with SpaCy
# lookups = Lookups()
# lookups.add_table('lemma_rules',{'noun':[['s','']]}) 
# lemmatizer = Lemmatizer(lookups)

'''

In SpaCy v3, the need for explicit use of Lookups has been minimized because lemmatization and other linguistic 
features are now integrated into the Doc object during processing
'''
for word in words_l:
    doc = nlp(word)
    # Lemmatization is already performed during processing, but you can still use your custom function
    lemma = Lemmatizer(doc[0])
    print(f'Lemmatization with SpaCy: {lemma}')

# Implementation of Lemmatization with NLTK
lemmatizer = WordNetLemmatizer()                    # Instantiating the Lemmatization class

nltk.download('wordnet')

for word in words_l:
    lemma = lemmatizer.lemmatize(word)
    print(f'Lemmatization with NLTK: {lemma}')


'''
In this tutorial, I learned about the lemmatization and its simple implementation using SpaCy and NLTK.
Also, during this tutorial, I got stuck with SpaCy v3 as the use of lemmatizer is different from that of 
SpaCy v2.
'''