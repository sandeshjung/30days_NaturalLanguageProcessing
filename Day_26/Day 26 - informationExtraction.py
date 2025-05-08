'''
(Named entities and relations)
A typical sentence may contain several named entities of various types, such as geographic entities, organizations, people, political entities,
times, artifacts, events, and natural phonomena. And a sentence can contain several relations, too-facts about the relationship between the 
named entities in the sentence.

(A knowledge base)
If you have your chatbot run information extraction on a large corpus, such as Wikipedia, that corpus will produce facts about the world that can 
inform future chatbot behaviors and replies. Some chatbots record all the information they extract in knowledge base. Such a knowledge base can later be queried
to help your chatbot make informed decisions or inferences about the world.
Chatbots can also store knowledge about the current user 'session' or converstation. Knowledge that is relevant only to the current conversation is called 
'context'. This contextual knowledge can be stored in the same global knowledge base that supports the chatbot, or it can be stored in a separate knowledge base.
Context can include facts about the user, the chatroom or channel, or the weather and news for that moment in time. 
A collection of triplets is a knowledge graph. 

(Information extraction)
Information extraction is converting unstructured text into structured information stored in a knowledge ase or knowledge graph. 

(Regular patterns)
We need a pattern-matching algorithm that can identify sequences of characters or words that match the pattern so we can 'extract' them from a longer string
of text. 
'''

# Pattern hardcoded in Python
def find_greeting(s):
    '''Return greeting str (Hi, etc) if greeting pattern matches'''
    if s[0] == 'H':
        if s[:3] in ['Hi', 'Hi ', 'Hi,', 'Hi!']:
            return s[:2]
        elif s[:6] in ['Hello','Hello ', 'Hello,', 'Hello!']:
            return s[:5]
    elif s[0] == 'Y':
        if s[1] == 'o' and s[:3] in ['Yo', 'Yo,', 'Yo ', 'Yo!']:
            return s[:2]
    return None


find_greeting('Hi Mr. Sandesh!')
find_greeting('Hello, Rosa.')

# This was tedious way of programming a pattern matching algorithm instead use regelar expressions

'''
(Regular expressions)
Regular expressions are strings written in special computer language that you can use to specify algorithms. They are the pattern definition language of choice 
for many NLP problems involving pattern matching. Regular expressions define a finite state machine or FSM-a tree of 'if-then' decisions about a sequence
of symbols, such as the find_greeting() function. 
Grammer refers to set of rules that determine whether or not a sequence of symbols is a valid member of a language, often called computer language or formal
language. 
eg. r'.\*'
'''

# Regular expression for GPS coordinates
import re
lat = r'([-]?[0-9]?[0-9][.][0-9]{2,10})'
lon = r'([-]?1?[0-9]?[0-9][.][0-9]{2,10})'
sep = r'[,/ ]{1,3}'
re_gps = re.compile(lat + sep + lon)
re_gps.findall('http://...maps/@34.0551066, -118.2496763...')
re_gps.findall("https://www.openstreetmap.org/#map=10/5.9666/116.0566")

# Regular expression for us dates
us = r'((([01]?\d)[-/]([0123]?\d))([-/]([0123]\d)\d\d)?)'
mdy = re.findall(us, 'Santa came 12/25/2017. An elf appeared 12/12.')
mdy

'''
Extracting relationships (relations)
The pattern we are going to use to extract relationships (or relations) is a pattern such as SUBJECT - VERB - OBJECT. To recognize these patterns, you'll
need your NLP pipeline to know the parts of speech for each word in a sentence.
'''

# POS tagging with spaCy
import spacy
en_model = spacy.load('en_core_web_md')
sentence = ("In 1541 Desoto wrote in his journal that the Pascagoula people " + "ranged as far north as the confluence of the Leaf and Chickasawhay rivers at 30.4, --88.5.")
parsed_sent = en_model(sentence)
parsed_sent.ents
' '.join(['{}_{}'.format(tok, tok.tag_) for tok in parsed_sent])


# Visualize a dependency tree
from spacy.displacy import render
sentence = "In 1541 Desoto wrote in his journal about the Pascagoula."
parsed_sent = en_model(sentence)
with open('pascagoula.html', 'w') as f:
    f.write(render(docs=parsed_sent, page=True, options=dict(compact=True)))


# Helper functions for spaCy tagged strings
import pandas as pd 
from collections import OrderedDict
def token_dict(token):
    return  OrderedDict(ORTH=token.orth_, LEMMA=token.lemma_,  POS=token.pos_, TAG=token.tag_, DEP=token.dep_)

def doc_dataframe(doc):
    return pd.DataFrame([token_dict(tok) for tok in doc])

doc_dataframe(en_model("In 1541 Desoto met the Pascagoula."))


'''
Entity name normalization
    The normalized representation of an entity is usually a string, even for numerical information such as dates. THe normalized ISO format for this date
    would be '1541-01-01'. A normalized representation for entities enables your knowledge base to connect all the different things that happened in the 
    world on that same date to that same node (entity) in your graph.
    Normalizing named entities and resolving ambiguities is often called coreference resolution or anaphora resolution, especially for pronouns or
    other 'names' relying on context. Normalization of named entities ensures that spelling and naming variations don't pollute your vocabulary of entity
    names with confounding, redundant names. 
'''


'''
Segmentation
    Document 'chunking' is useful for creating semi-structured data about documents that can make it easier to search, filter, and sort documents for 
    information retrieval. When you divide natural language text into meaningful pieces, it's called segmentation. The resulting segments can be phrases,
    sentences, quotes, paragraphs, or even entire sections of a long document.
    Sentences are the most common chunk for most information extraction problems. Grammatically correct English language sentences must contain a subject (noun)
    and a verb, which means they'll usually have at least one relation or fact worth extracting. 
    For chatbot pipeline, your goal is to segment documents into sentences, or statements.

Sentence Segmentation
    It is usually the first step in an information extraction pipeline. It helps isolate facts from each other so that you can associate the right price with
    the right thing in a string. Sentences contain a logically cohesive statement about the world. 
    Manually programmed algorithms and statitical models
'''

# Sentence segmentation with regular expressions
import re
import string
re.split(r'[!.?]+[ $]', "Hello World.... Are you there?!?! I'm going to Mars!")
re.split(r'(?<!\d)\.|\.(?!\d)', "I went to GT.You?")



'''
Summary
    A knowledge graph can be built to store relationships between entities.
    Regular expressions are mini-programming language that can isolate and extract information.
    POS tagging allows you to extract relationships between entities mentioned in a sentence.
    Segmenting sentences requires more that just splitting on periods and exclamation marks.
'''