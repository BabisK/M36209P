import random
from itertools import chain
from collections import defaultdict
from nltk import downloader, ngrams, FreqDist, download
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import EuroparlCorpusReader
from nltk.probability import LaplaceProbDist, ConditionalFreqDist, ConditionalProbDist
from nltk.tokenize import word_tokenize
import matplotlib


def predict_word(probability_distribution, sentence, model):
    tokens = word_tokenize(sentence)
    tokens = ['start1', 'start0'] + tokens
    gram = tuple()
    if model == 'bigram':
        gram = tuple(tokens[-1:], )
    elif model == 'trigram':
        gram = tuple(tokens[-2:])
    pd = [word[0] for word in probability_distribution[gram].freqdist().most_common() if word[0] != '<unknown>']
    return (pd[0] if len(pd) > 0 else None)


def centropy_perplexity(probability_distribution, ngrams):
    logprob = [probability_distribution[tuple(ngram[:-1])].logprob(ngram[-1]) for ngram in ngrams]
    logprob = sum(logprob)
    cross_entropy = -logprob / len(ngrams)
    perplexity = 2 ** cross_entropy
    return cross_entropy, perplexity


def ngrams_sentences(sentences, n):
    ngrams_serntences = []
    for sentence in sentences:
        for i in range(n-1):
            sentence = ['start{}'.format(i)] + sentence + ['end{}'.format(i)]
        ngrams_serntences.append([gram for gram in ngrams(sentence, n)])
    return ngrams_serntences


def main():
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    download('punkt')
    # Download and load the english europarl corpus
    downloader.download('europarl_raw')
    english = LazyCorpusLoader('europarl_raw/english', EuroparlCorpusReader, r'ep-.*\.en', encoding='utf-8')

    words = english.words()

    # Calculate the frequency distribution of the words in the corpus
    word_frequency_distribution = FreqDist([word.lower() for word in words])

    # Get the sentences of the corpus, all in lower case, with infrequent words replaced by the token "<unknown>"
    sentences = [[word.lower() if word_frequency_distribution[word.lower()] >= 10 else '<unknown>' for word in sentence]
                 for sentence in english.sents()]

    # create train and test dataset
    train = sentences[0:int(len(sentences) * 0.8)]
    test = sentences[int(len(sentences) * 0.8):]

    vocabulary_length = word_frequency_distribution.B()

    # Calculate bigrams and trigrams
    bigrams_train = list(chain.from_iterable(ngrams_sentences(train, 2)))
    trigrams_train = list(chain.from_iterable(ngrams_sentences(train, 3)))

    # Calculate the conditional frequency distributions for bigrams and trigrams
    bigrams_fd = ConditionalFreqDist(((f,), s) for f, s in bigrams_train)
    trigrams_fd = ConditionalFreqDist([((f, s), t) for f, s, t in trigrams_train])

    # Calculate the conditional probability distributions for bigrams and trigrams
    cpd_bigram = ConditionalProbDist(bigrams_fd, LaplaceProbDist, vocabulary_length)
    cpd_trigram = ConditionalProbDist(trigrams_fd, LaplaceProbDist, vocabulary_length)

    bigrams_test = ngrams_sentences(test, 2)
    bigram_length_probabilities = defaultdict(list)
    for sentence in bigrams_test:
        logprob = [cpd_bigram[(w1,)].logprob(w2) for w1, w2 in sentence]
        logprob = sum(logprob)
        bigram_length_probabilities[len(sentence)].append(logprob)

    x = 0
    s = None
    for sentence in bigrams_test:
        if (len(sentence) > x):
            x = len(sentence)
            s = sentence

    trigrams_test = ngrams_sentences(test, 3)
    trigram_length_probabilities = defaultdict(list)
    for sentence in trigrams_test:
        logprob = [cpd_trigram[(w1, w2)].logprob(w3) for w1, w2, w3 in sentence]
        logprob = sum(logprob)
        trigram_length_probabilities[len(sentence)].append(logprob)

    average_bigram_length_probabilities = {
        length: sum(bigram_length_probabilities[length]) / float(len(bigram_length_probabilities[length])) for length in
        bigram_length_probabilities.keys()}
    average_trigram_length_probabilities = {
        length: sum(trigram_length_probabilities[length]) / float(len(trigram_length_probabilities[length])) for length
        in
        trigram_length_probabilities.keys()}

    random_sentences = [[words[random.randint(0, len(words) - 1)].lower() for i in range(key)] for key in
                        bigram_length_probabilities.keys()]

    bigrams_random = ngrams_sentences(random_sentences, 2)
    random_bigram_length_probabilities = defaultdict(list)
    for sentence in bigrams_random:
        logprob = [cpd_trigram[(w1,)].logprob(w2) for w1, w2 in sentence]
        logprob = sum(logprob)
        random_bigram_length_probabilities[len(sentence)].append(logprob)

    trigrams_random = ngrams_sentences(random_sentences, 3)
    random_trigram_length_probabilities = defaultdict(list)
    for sentence in trigrams_random:
        logprob = [cpd_trigram[(w1, w2)].logprob(w3) for w1, w2, w3 in sentence]
        logprob = sum(logprob)
        random_trigram_length_probabilities[len(sentence)].append(logprob)

    bigram = plt.scatter(list(average_bigram_length_probabilities.values()),
                         list(average_bigram_length_probabilities.keys()), color='red')
    trigram = plt.scatter(list(average_trigram_length_probabilities.values()),
                          list(average_trigram_length_probabilities.keys()), color='blue')
    random_bigram = plt.scatter(list(random_bigram_length_probabilities.values()),
                                list(random_bigram_length_probabilities.keys()), color='green')
    random_trigram = plt.scatter(list(random_trigram_length_probabilities.values()),
                                 list(random_trigram_length_probabilities.keys()), color='black')
    plt.xlabel('$log_2(P(W_1^k))$')
    plt.ylabel('$k$')
    plt.legend((bigram, trigram, random_bigram, random_trigram),
               ('Bigram', 'Trigram', 'Random bigram', 'Random trigram'))
    plt.ylim(ymin=0)
    #plt.show()
    plt.savefig('logprob')

    seed = 'this'
    for i in range(10):
        newword = predict_word(cpd_bigram, seed, 'bigram')
        if newword != None:
            seed += ' ' + newword
        else:
            break
    print('Given the seed word "this", the bigram model produced this text of length 10: {}'.format(seed))

    seed = 'this'
    for i in range(10):
        newword = predict_word(cpd_trigram, seed, 'trigram')
        if newword != None:
            seed += ' ' + newword
        else:
            break
    print('Given the seed word "this", the trigram model produced this text of length 10: {}'.format(seed))

    test_bigrams = []
    for sentence in bigrams_test:
        test_bigrams += sentence
    bigram_entropy, bigram_perplexity = centropy_perplexity(cpd_bigram, test_bigrams)
    print('Cross-entropy of the bigram model is {}. The corresponding perplexity is {}'.format(bigram_entropy,
                                                                                               bigram_perplexity))

    test_trigrams = []
    for sentence in trigrams_test:
        test_trigrams += sentence
    trigram_entropy, trigram_perplexity = centropy_perplexity(cpd_trigram, test_trigrams)
    print('Cross-entropy of the trigram model is {}. The corresponding perplexity is {}'.format(trigram_entropy,
                                                                                                trigram_perplexity))


if __name__ == "__main__":
    main()
