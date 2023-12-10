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