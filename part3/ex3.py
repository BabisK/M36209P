import random
from itertools import chain
from collections import defaultdict
from nltk import downloader, ngrams, FreqDist, download
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import EuroparlCorpusReader
from nltk.probability import LaplaceProbDist, ConditionalFreqDist, ConditionalProbDist
from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize
import numpy
import matplotlib
import string
import operator
import copy

def ngrams_sentences(sentences, n):
    ngrams_serntences = []
    for sentence in sentences:
        for i in range(n - 1):
            sentence = ['start{}'.format(i)] + sentence + ['end{}'.format(i)]
        ngrams_serntences.append([gram for gram in ngrams(sentence, n)])
    return ngrams_serntences


def word_emited_probability(emited, actual):
    d = edit_distance(emited, actual)
    return 1 - (float(edit_distance(emited, actual)) / (max(len(emited), len(actual)) + 1))


def viterbi(sentence, vocabulary, conditional_probability_distribution):
    V = [{}]
    for word in vocabulary:
        V[0][word] = {
            'prob': conditional_probability_distribution['start0'].prob(word) * word_emited_probability(sentence[0], word),
            'prev': None}

    for i, token in enumerate(sentence[1:], start=1):
        V.append({})
        for word in vocabulary:
            prob = [V[i - 1][prev_word]['prob'] * conditional_probability_distribution[prev_word].prob(word) for prev_word in vocabulary]
            index, max_prob = max(enumerate(prob), key=operator.itemgetter(1))
            max_prob = max_prob * word_emited_probability(word, token)
            V[i][word] = {'prob': max_prob, 'prev': vocabulary[index]}

    opt = []
    max_prob = max(value['prob'] for value in V[-1].values())
    previous = None
    for word, data in V[-1].items():
        if data['prob'] == max_prob:
            opt.append(word)
            previous = word
            break

    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]['prev'])
        previous = V[t + 1][previous]['prev']
    return opt


def main():
    # matplotlib.use('Qt5Agg')
    # import matplotlib.pyplot as plt

    download('punkt')
    # Download and load the english europarl corpus
    downloader.download('europarl_raw')
    english = LazyCorpusLoader('europarl_raw/english', EuroparlCorpusReader, r'ep-.*\.en', encoding='utf-8')

    words = english.words()

    # Calculate the frequency distribution of the words in the corpus
    word_frequency_distribution = FreqDist([word.lower() for word in words])

    # Get the sentences of the corpus, all in lower case, with infrequent words replaced by the token "<unknown>"
    sentences = [
        ['start0'] + [word.lower() if word_frequency_distribution[word.lower()] >= 10 else '<unknown>' for word in
                      sentence] + ['end0']
        for sentence in english.sents()]

    # create train and test dataset
    train = sentences[0:int(len(sentences) * 0.8)]
    test = sentences[int(len(sentences) * 0.8):]

    vocabulary = list(word_frequency_distribution)
    vocabulary_length = word_frequency_distribution.B()

    # Calculate bigrams
    bigrams_train = list(chain.from_iterable(ngrams_sentences(train, 2)))

    # Calculate the conditional frequency distribution for bigrams
    bigrams_fd = ConditionalFreqDist(((f,), s) for f, s in bigrams_train)

    # Calculate the conditional probability distribution for bigrams
    cpd_bigram = ConditionalProbDist(bigrams_fd, LaplaceProbDist, vocabulary_length)

    lower_case_letters = string.ascii_lowercase
    error_test = copy.deepcopy(test)
    for sentence in error_test:
        word = random.randrange(1, len(sentence)-1)
        sentence[word] = random.choice(vocabulary)
        word = random.choice(sentence[1:-2])
        word = random.randrange(1, len(sentence) - 1)
        letter = random.randrange(0, len(sentence[word]))
        sentence[word] = sentence[word][0:letter] + random.choice(lower_case_letters) + sentence[word][letter+1:]

    corrected = viterbi(error_test[25][:-1], vocabulary, cpd_bigram)

    print('Corrected:{}'.format(corrected))
    print('Original:{}'.format(test[25]))

if __name__ == "__main__":
    main()
