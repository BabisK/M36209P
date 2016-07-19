import re
from nltk import downloader, ngrams, FreqDist
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import EuroparlCorpusReader
from nltk.probability import LaplaceProbDist
from pprint import pprint

def main():
    downloader.download('europarl_raw')
    english = LazyCorpusLoader('europarl_raw/english', EuroparlCorpusReader, r'ep-.*\.en', encoding='utf-8')
    tokens = english.words()

    bigrams = [ gram for gram in ngrams(tokens, 2) ]
    trigrams = [ gram for gram in ngrams(tokens, 3) ]

    counts = FreqDist(bigrams)
        
    #laplace = LaplaceProbDist(counts)

    pprint(counts.N())

    #print(len(set(english.words())))

if __name__ == "__main__":
    main()
