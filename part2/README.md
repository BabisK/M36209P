#Exercises on text classification
######Ion Androutsopoulos, 2015–16

17. Develop a text classifier for a kind of texts of your choice (e.g., e-mail messages, tweets, customer reviews) and at least two classes (e.g., 
spam/ham, positive/negative/neutral). You should write your own code to convert each (training, validation, or test) text to a feature vector. You may 
use Boolean, TF, or TF-IDF features corresponding to words or n-grams, to which you can also add other features (e.g., length of the text). You may 
apply any feature selection (or dimensionality reduction) method you consider appropriate. You may also want to try using centroids of pre-trained word 
embeddings (slide 35). You can write your own code to perform feature selection (or dimensionality reduction) and to train the classifier (e.g., using 
SGD and the tricks of slides 58 and 59, in the case of logistic regression), or you can use existing implementations. 5 You should experiment with at 
least logistic regression, and optionally other learning algorithms (e.g., Naive Bayes, k-NN, SVM). Draw learning curves (slides 62, 65) with 
appropriate measures (e.g., accuracy, F1) and precision-recall curves (slide 23). Include experimental results of appropriate baselines (e.g., majority 
classifiers). Make sure that you use separate training and test data. Tune the feature set and hyper-parameters (e.g., regularization weight λ) on a 
held-out part of the training data or using a cross-validation (slide 25) on the training data. Document clearly in a short report (max. 10 pages) how 
your system works and its experimental results.
