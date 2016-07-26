import random
from itertools import chain
from collections import defaultdict
from nltk import downloader, ngrams, FreqDist, download
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import EuroparlCorpusReader
from nltk.probability import LaplaceProbDist, ConditionalFreqDist, ConditionalProbDist
from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize
import matplotlib

def ngrams_sentences(sentences, n):
    ngrams_serntences = []
    for sentence in sentences:
        for i in range(n - 1):
            sentence = ['start{}'.format(i)] + sentence + ['end{}'.format(i)]
        ngrams_serntences.append([gram for gram in ngrams(sentence, n)])
    return ngrams_serntences

def word_emited_probability(emited, actual):
    d = edit_distance(emited, actual)
    return 1 - (float(edit_distance(emited, actual))/(max(len(emited), len(actual))+1))

def viterbi(sentence, vocabulary, conditional_probability_distribution):
    V = [{}]
    for word in vocabulary:
        p1 = conditional_probability_distribution[sentence[0]].prob(word)
        p2 = word_emited_probability(sentence[0], word)
        d = edit_distance(u"test", u"rest")
        V[0][word] = {'prob': conditional_probability_distribution[sentence[0]].prob(word) * word_emited_probability(sentence[0], word)}
    return V
    '''
    new_sentences, vk1 = viterbi(sentence[:-1], conditional_probability_distribution)
    v = [conditional_probability_distribution[new_sentence[-1]].max() for new_sentence in new_sentences]
    return new_sentence, v
    '''

def main():
    #matplotlib.use('Qt5Agg')
    #import matplotlib.pyplot as plt

    download('punkt')
    # Download and load the english europarl corpus
    downloader.download('europarl_raw')
    english = LazyCorpusLoader('europarl_raw/english', EuroparlCorpusReader, r'ep-.*\.en', encoding='utf-8')

    words = english.words()

    vocabulary = set(words)

    # Calculate the frequency distribution of the words in the corpus
    word_frequency_distribution = FreqDist([word.lower() for word in words])

    # Get the sentences of the corpus, all in lower case, with infrequent words replaced by the token "<unknown>"
    sentences = [[word.lower() if word_frequency_distribution[word.lower()] >= 10 else '<unknown>' for word in sentence]
                 for sentence in english.sents()]

    # create train and test dataset
    train = sentences[0:int(len(sentences) * 0.8)]
    test = sentences[int(len(sentences) * 0.8):]

    vocabulary_length = word_frequency_distribution.B()

    # Calculate bigrams
    bigrams_train = list(chain.from_iterable(ngrams_sentences(train, 2)))

    # Calculate the conditional frequency distribution for bigrams
    bigrams_fd = ConditionalFreqDist(((f,), s) for f, s in bigrams_train)

    # Calculate the conditional probability distribution for bigrams
    cpd_bigram = ConditionalProbDist(bigrams_fd, LaplaceProbDist, vocabulary_length)

    viterbi(test[0], vocabulary, cpd_bigram)

if __name__ == "__main__":
    main()