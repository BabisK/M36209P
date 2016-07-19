import re
from nltk import downloader, ngrams, FreqDist
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import EuroparlCorpusReader
from nltk.probability import LaplaceProbDist, ConditionalFreqDist, ConditionalProbDist
from pprint import pprint


def main():
    # Download and load the english europarl corpus
    downloader.download('europarl_raw')
    english = LazyCorpusLoader('europarl_raw/english', EuroparlCorpusReader, r'ep-.*\.en', encoding='utf-8')

    # Get the words of the corpus, all in lower case
    tokens = [token for token in english.words()]
    # Find the uncommon words (frequency less than 10)
    uncommon_words = [word for word, freq in FreqDist(tokens).items() if freq < 10]
    # Replace the uncommon words with the pseudoword "<unknown>"
    tokens = [word if word not in uncommon_words else "<unknown>" for word in tokens]

    # create train and test dataset
    train = tokens[0:int(len(tokens) * 0.8)]
    test = tokens[int(len(tokens) * 0.8):]

    # Calculate bigrams and trigrams
    bigrams = [gram for gram in ngrams(train, 2)]
    trigrams = [gram for gram in ngrams(train, 3)]

    # Calculate models using laplace smoothing based on bigrams and trigrams
    #bigrams_laplace = LaplaceProbDist(FreqDist(bigrams))
    #trigrams_laplace = LaplaceProbDist(FreqDist(trigrams))
    bigrams_laplace = ConditionalFreqDist(bigrams)
    trigrams_laplace = ConditionalFreqDist([((f,s), t) for f,s,t in trigrams])

    cpd = ConditionalProbDist(bigrams_laplace, LaplaceProbDist)

    start, end = -1
    for i, t in enumerate(test):
        if t == ".":
            if start == -1:
                start = i + 1
            else:
                end = i

    test_bigrams = [gram for gram in ngrams(test[start:end], 2)]
    test_trigrams = [gram for gram in ngrams(test[start:end], 3)]

    print(len(set(english.words())))


if __name__ == "__main__":
    main()
