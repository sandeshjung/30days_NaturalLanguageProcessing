import numpy as np
import matplotlib.pyplot as plt
import nltk

colors = 'rgbcmyk'

def bar_chart(categories, words, counts):
    '''
    Plot a bar chart showing counts for each word by category
    '''
    ind = np.arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = plt.bar(ind+c*width, counts[categories[c]], width,
                         color=colors[c % len(colors)])
        bar_groups.append(bars)
    plt.xticks(ind+width, words)
    plt.legend([b[0] for b in bar_groups], categories, loc='upper left')
    plt.ylabel('Frequency')
    plt.title('Frequency of Six Modal Verbs by Genre')
    plt.show()

genres = ['news','religion','hobbies','government','adventure']
modals = ['can','could','may','might','must','will']
cfdist = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in genres
    for word in nltk.corpus.brown.words(categories=genre)
    if word in modals
)
counts = {}
for genre in genres:
    counts[genre] = [cfdist[genre][word] for word in modals]
bar_chart(genres, modals, counts)

'''
In this tutorial, I learned how to use matplotlib for NLP for frequency analysis. 
Also, I got stuck with matplotlib package error for hours.
'''