'''
This pattern matching chatbot is an example of a tightly controlled chatbot. Pattern matching chatbots were
common before modern machine learning chatbot techniques were developed. 
Let's build an FSM, a regular expression, that can speak lock language(regular langues). 
'''
import re
r = "{hi|hello|hey}[ ]*([a-z]*)"                # | means 'or' and '\*' means the preceding character can occur 0 or more times and still match
# So our regex will match greetings that start with hi or hello or hey followed by an number of <space> characters and then any no of letters.
re.match(r, 'Hello Sandesh', flags=re.IGNORECASE)  # Ignoring case of text characters is common, to keep regular expressions simpler
r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|"\
    r"afternoon|even[gin']{0,3}))[\s,:;]{1,3}([a-z]{1,20})"         # Complex pattern if Regular expression
re_greeting = re.compile(r, flags=re.IGNORECASE)            # Compile helps to ignore flags parameters
re_greeting.match("Good Morning, Sandesh")
re_greeting.match("Good Morning, Sandesh").groups()


# Output Generator for chatbot
my_names = set(['sandesh', 'jung', 'chatty','chatbot'])
curt_names = set(['hal','you','u'])
greeter_name = '' # We dont yet know who is chatting with bot
match = re_greeting.match(input())
if match:
    name = match.groups()[-1]
    if name in curt_names:
        print("Good One.")
    elif name.lower() in my_names:
        print(f"Hi {greeter_name}, How are you?")


# Sorting Tokens
from collections import Counter
Counter('There is a dog infront of my house. These is a brown dog.'.split())        # Counter dictionary bins the objects and counts them
# Word order and permutations
from itertools import permutations
[" ".join(n) for n in \
 permutations("Good Morning, Sandesh!".split(), 3)]             # Ordering of the words in sequence



'''
Today I have read and implemented the First Chapter of book Natural Language Processing in Action.
I covered NLP and the magic, Regular Expressions, Finite State Machine or FSM concept, Word order and Grammer,
simples NLP chatbot using Regular expressions and FSM concept. 
'''