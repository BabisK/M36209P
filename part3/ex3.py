import random
from itertools import chain
from collections import defaultdict
from nltk import downloader, ngrams, FreqDist, download
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import EuroparlCorpusReader
from nltk.probability import LaplaceProbDist, ConditionalFreqDist, ConditionalProbDist
from nltk.tokenize import word_tokenize
import matplotlib

