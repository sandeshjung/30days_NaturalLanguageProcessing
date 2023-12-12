'''
.	        Wildcard, matches any character
^abc	    Matches some pattern abc at the start of a string
abc$	    Matches some pattern abc at the end of a string
[abc]	    Matches one of a set of characters
[A-Z0-9]	Matches one of a range of characters
ed|ing|s	Matches one of the specified strings (disjunction)
*	        Zero or more of previous item, e.g. a*, [a-z]* (also known as Kleene Closure)
+	        One or more of previous item, e.g. a+, [a-z]+
?	        Zero or one of the previous item (i.e. optional), e.g. a?, [a-z]?
{n}	        Exactly n repeats where n is a non-negative integer
{n,}	    At least n repeats
{,n}	    No more than n repeats
{m,n}	    At least m and no more than n repeats
a(b|c)+	    Parentheses that indicate the scope of the operators
'''

import re
import nltk

nltk.download('words')
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]                    #using word corpus


# Using Basic Meta-Characters
[w for w in wordlist if re.search('ed$', w)]                    # find words ending with ed using regular expression <<ed$>>
# We use the re.search(p,s) function to check where the pattern p can be found in string s
[w for w in wordlist if re.search('^..j..t..$', w)]             # . wildcard matches any single character
[w for w in wordlist if re.search('..j..t..', w)]               # caret symbol(^) matches characters of start of word
[w for w in wordlist if re.search('^e-?mail$', w)]              # ? specifies the prev character is optional


# Ranges and Closures
[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]
'''
T9 system is used for entering text on mobile phones. Two or more words that are entered with the same
sequence of keystrokes are called textonyms.
eg. hole and golf both are entered by pressing sequence 4653.
'''

nltk.download('nps_chat')
chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
[w for w in chat_words if re.search('^m+i+n+e+$',w)]            # + means one or more instances of preceding item
[w for w in chat_words if re.search('^[ha]+$',w)] 
[w for w in chat_words if re.search('^m*i*n*e*$', w)]           # * means zero or more instances of preceding item


nltk.download('treebank')
wsj = sorted(set(nltk.corpus.treebank.words()))
[w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]            #\ means the following character is deprived
[w for w in wsj if re.search('^[A-Z]+\$$', w)]  
[w for w in wsj if re.search('[0-9]{4}$', w)]                   # Braced expression specifies the no of repeats of previous items
[w for w in wsj if re.search('(ed|ing)$', w)]                   # | specifies the choice between left and right



# Applications of Regular Expressions
word = 'supercalifragilisticexpialidocious'
re.findall(r'[aeiou]', word)                                    # re.findall() method finds all non-overlapping matches of given regular expression
fd = nltk.FreqDist(vs for word in wsj
                   for vs in re.findall(r'[aeiou]{2,}', word))
fd.most_common(12)                                              # most common words obtained

# eg for initial vowel sequences, final vowel sequences, and all consonant; everything else is ignored
# if one of the three parts matches the word, any later parts of the reqular expression are ignored
regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def compress(word):
    pieces = re.findall(regexp, word)       # extract all the matching pieces
    return ''.join(pieces)                  # join them together

nltk.download('udhr')
english_udhr = nltk.corpus.udhr.words('English-Latin1')
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))

re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')
re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')       # ?: opposes the concept of parentheses i.e give output only if its inside
re.split(r'\s+', word)                      # \s means any whitespace character
re.findall(r'\w+|\s\w*', word)              # \w means alphanumerical characters


# NLTK's Regular Expression Tokenizer
'''
function nltk.regexp_tokenize() is similar to re.findall(). 
however, nltk.regexp_tokenize() is more efficient for this task, and avoids the need for special treatment of parantheses.
'''
text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'''(?x)          # set flag to allow verbose regexps, (strip out embedded whitespace and comments)
            (?:[A-Z]\.)+    # abbreviations, e.g. U.S.A.
            | \w+(?:-\w+)*  # words with optional internal hyphens
            | \$\d+(?:\.\d+)?%?   # currency and percentages, e.g. $12.40, 82%
            | \.\.\.        # ellipsis
            | [][.,;"'?():-_`]    # these are separate tokens; includes ], []

            '''

nltk.regexp_tokenize(text, pattern)
# while using the verbose flav, you can no longer use ' ' to match space character; use \s instead


'''
Word segmentation:
    For some writing systems, tokenizing text is made more difficult by the fact that there is no visual representation of word boundaries. 
    For example, in Chinese, the three-character string: 爱国人 (ai4 "love" (verb), guo2 "country", ren2 "person") could be tokenized as 
    爱国 / 人, "country-loving person" or as 爱 / 国人, "love country-person."
'''

def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i + 1
    words.append(text[last:])
    return words

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"

segment(text, seg1)
segment(text, seg2)