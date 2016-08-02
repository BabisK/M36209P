#Exercises on n-gram language models, spelling correction, text normalization
######Ion Androutsopoulos, 2015–16

1. Implement (in any programming language) a bigram and a trigram language model (for word sequences), using Laplace smoothing (or better).
2. Train your models on a subset of a corpus (e.g., from the English or Greek part of Europarl), after replacing (both in the training and the test 
subset) all the out-of-vocabulary words (e.g., words that do not occur at least 10 times in the training subset) by a special token (e.g., *unknown*).
3. Check the log-probabilities that your trained models return when given (correct) sentences from the test subset vs. (incorrect) sentences of the same 
length (in words) consisting of randomly selected vocabulary words.
4. Demonstrate how your models could predict the next (vocabulary) word, as in a predictive keyboard (slide 31, center).
5. Estimate the language cross-entropy and perplexity of your models on a test subset of the corpus.
6. Optionally combine your two models using linear interpolation (slide 13) and check if the combined model performs better. You are allowed to use NLTK 
(http://www.nltk.org/) or other tools for sentence splitting, tokenization, and counting n-grams, but otherwise you should write your own code.
