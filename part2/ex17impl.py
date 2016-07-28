import pandas
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, log_loss
from sklearn.learning_curve import learning_curve
import matplotlib

def logistic_clf(data, target):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('tsvd', TruncatedSVD()),
                         ('clf', LogisticRegression()),
                         ])

    parameters = {'vect__ngram_range': [(1, 2),],
                  #'vect__stop_words': (None, 'english'),
                  #'vect__lowercase': (True, False),
                  #'tfidf__use_idf': (True, False),
                  'tsvd__n_components': (1100, ),
                  #'clf__penalty': ('l1', 'l2'),
                  'clf__C': [0.1, 1]
                  }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf.fit(data, target)

    return gs_clf

def tree_clf(data, target):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('tsvd', TruncatedSVD()),
                         ('clf', ExtraTreeClassifier()),
                         ])

    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                  'vect__stop_words': (None, 'english'),
                  'vect__lowercase': (True, False),
                  'tfidf__use_idf': (True, False),
                  'tsvd__n_components': (500, 700, 900, 1100, 1300, 1500),
                  }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    #gs_clf.fit(data, target)

    return gs_clf

def svm_clf(data, target):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('tsvd', TruncatedSVD()),
                         ('clf', SVC()),
                         ])

    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                  'vect__stop_words': (None, 'english'),
                  'vect__lowercase': (True, False),
                  'tfidf__use_idf': (True, False),
                  'tsvd__n_components': (500, 700, 900, 1100, 1300, 1500),
                  'clf__penalty': ('l1', 'l2'),
                  }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf.fit(data, target)

    return gs_clf

def main():
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    train_df = pandas.read_csv('./train.csv', index_col=0,engine='python')
    train_df = train_df.dropna()

    dev_df = pandas.read_csv('./dev.csv', index_col=0, engine='python')
    dev_df = dev_df.dropna()

    devtest_df = pandas.read_csv('./devtest.csv', index_col=0, engine='python')
    devtest_df = devtest_df.dropna()
    '''
    logisticclf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('tsvd', TruncatedSVD()),
                         ('clf', LogisticRegression()),
                         ])

    parameters = {'vect__ngram_range': [(1, 2)],
                  #'vect__stop_words': (None, 'english'),
                  #'vect__lowercase': (True, False),
                  #'tfidf__use_idf': (True, False),
                  'tsvd__n_components': (1100, ),
                  #'clf__penalty': ('l1', 'l2'),
                  'clf__C': [1]
                  }

    p = ParameterGrid(parameters)
    logisticclf.set_params(**p[0])

    train_size, train_score, valid_score = learning_curve(logisticclf, train_df.tweet, train_df.sentiment)
    ts = [sum(row)/len(row) for row in train_score]
    vs = [sum(row) / len(row) for row in valid_score]
    plt.figure()
    plt.title('Learning curve for Logistic Regression')
    plt.plot(train_size, ts, lw=2, label='cross-validation error')
    plt.plot(train_size, vs, lw=2, label='training error')
    plt.legend()
    plt.xlabel('Training samples')
    plt.ylabel('Error')
    plt.savefig('logreg.png')
    logisticclf.fit(train_df.tweet, train_df.sentiment)
    '''

    treeclf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('tsvd', TruncatedSVD()),
                         ('clf', ExtraTreeClassifier()),
                         ])

    parameters = {'vect__ngram_range': [(1, 2)],
                  #'vect__stop_words': (None, 'english'),
                  #'vect__lowercase': (True, False),
                  #'tfidf__use_idf': (True, False),
                  'tsvd__n_components': (1100,),
                  }
    p = ParameterGrid(parameters)
    treeclf.set_params(**p[0])

    train_size, train_score, valid_score = learning_curve(treeclf, train_df.tweet, train_df.sentiment)
    ts = [sum(row) / len(row) for row in train_score]
    vs = [sum(row) / len(row) for row in valid_score]
    plt.figure()
    plt.title('Learning curve for Extra Tree')
    plt.plot(train_size, ts, lw=2, label='cross-validation error')
    plt.plot(train_size, vs, lw=2, label='training error')
    plt.legend()
    plt.xlabel('Training samples')
    plt.ylabel('Error')
    plt.savefig('tree.png')
    treeclf.fit(train_df.tweet, train_df.sentiment)


    svmclf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('tsvd', TruncatedSVD()),
                         ('clf', SVC()),
                         ])

    parameters = {'vect__ngram_range': [(1, 2)],
                  #'vect__stop_words': (None, 'english'),
                  #'vect__lowercase': (True, False),
                  #'tfidf__use_idf': (True, False),
                  'tsvd__n_components': (1100,),
                  #'clf__penalty': ('l1', 'l2'),
                  'clf__probability': (True,)
                  }
    p = ParameterGrid(parameters)
    svmclf.set_params(**p[0])

    train_size, train_score, valid_score = learning_curve(svmclf, train_df.tweet, train_df.sentiment)
    ts = [sum(row) / len(row) for row in train_score]
    vs = [sum(row) / len(row) for row in valid_score]
    plt.figure()
    plt.title('Learning curve for SVM')
    plt.plot(train_size, ts, lw=2, label='cross-validation error')
    plt.plot(train_size, vs, lw=2, label='training error')
    plt.legend()
    plt.xlabel('Training samples')
    plt.ylabel('Error')
    plt.savefig('svm.png')
    svmclf.fit(train_df.tweet, train_df.sentiment)


    logreg_predicted_test = logisticclf.predict(dev_df.tweet)
    logreg_proba_test = logisticclf.predict_proba(dev_df.tweet)
    logreg_acc = accuracy_score(dev_df.sentiment, logreg_predicted_test)
    logreg_scores = f1_score(dev_df.sentiment, logreg_predicted_test, average='micro')
    logreg_loss = log_loss(dev_df.sentiment, logreg_proba_test)

    tree_predicted_test = treeclf.predict(dev_df.tweet)
    tree_proba_test = treeclf.predict_proba(dev_df.tweet)
    tree_acc = accuracy_score(dev_df.sentiment, tree_predicted_test)
    tree_scores = f1_score(dev_df.sentiment, tree_predicted_test, average='micro')
    tree_loss = log_loss(dev_df.sentiment, tree_proba_test)

    svm_predicted_test = svmclf.predict(dev_df.tweet)
    svm_proba_test = svmclf.predict_proba(dev_df.tweet)
    svm_acc = accuracy_score(dev_df.sentiment, svm_predicted_test)
    svm_scores = f1_score(dev_df.sentiment, svm_predicted_test, average='micro')
    svm_loss = log_loss(dev_df.sentiment, svm_proba_test)

    plt.figure()
    plt.title('Accuracy')
    plt.bar(0, logreg_acc, 0.50, color = 'blue', label='Logistic Regression')
    plt.bar(1, tree_acc, 0.50, color = 'red', label='Extra Tree')
    plt.bar(2, svm_acc, 0.50, color='green', label='SVM')
    plt.legend(loc=4)
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy.png')

    plt.figure()
    plt.title('F1')
    plt.bar(0, logreg_scores, 0.50, color='blue', label='Logistic Regression')
    plt.bar(1, tree_scores, 0.50, color='red', label='Extra Tree')
    plt.bar(2, svm_scores, 0.50, color='green', label='SVM')
    plt.legend(loc=4)
    plt.xlabel('Classifiers')
    plt.ylabel('F1')
    plt.savefig('f1.png')

if __name__ == "__main__":
    main()