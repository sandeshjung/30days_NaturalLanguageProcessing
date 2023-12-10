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