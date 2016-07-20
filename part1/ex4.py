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

    # Calculate the conditional frequency distributions for bigrams and trigrams
    #bigrams_laplace = LaplaceProbDist(FreqDist(bigrams))
    #trigrams_laplace = LaplaceProbDist(FreqDist(trigrams))
    bigrams_fd = ConditionalFreqDist(bigrams)
    trigrams_fd = ConditionalFreqDist([((f,s), t) for f,s,t in trigrams])

    # Calculate the conditional probability distributions for bigrams and trigrams
    cpd_bigram = ConditionalProbDist(bigrams_fd, LaplaceProbDist)
    cpd_trigram = ConditionalProbDist(trigrams_fd, LaplaceProbDist)

    start, end = -1, -1
    for i, t in enumerate(train):
        if t == ".":
            if start == -1:
                start = i + 1
            else:
                end = i
                break

    test_bigrams = [gram for gram in ngrams(test[start:end], 2)]
    test_trigrams = [gram for gram in ngrams(test[start:end], 3)]

    logprob_bi = [cpd_bigram[w1].logprob(w2) for w1, w2 in test_bigrams]
    logprob_bi = sum(logprob_bi)

    print(2**logprob_bi)


if __name__ == "__main__":
    main()
