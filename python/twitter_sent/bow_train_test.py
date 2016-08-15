#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

classes = ['0','1']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, weight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, self.classes, rotation=45)
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label', weight='bold')
    plt.xlabel('Predicted label', weight='bold')


def show_confusion_matrix(l_test, predicted):
    # Compute confusion matrix
    cm = metrics.confusion_matrix(l_test, predicted)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plot_confusion_matrix(cm)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print('Safety Success - Normalized confusion matrix')
    print(cm_normalized)
    plot_confusion_matrix(cm_normalized)
    plt.show()

data_path = '/home/dressag1/Documents/Datasets/imdb_movie/'
if __name__ == '__main__':

    print ('Loading the data')
    print("Loading data...")
    neg_train_path = '../data/twitter_train_100000_and_test.neg'
    pos_train_path = '../data/twitter_train_100000_and_test.pos'


    f_pos = open(pos_train_path, 'r')
    f_neg = open(neg_train_path,'r')

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    labels = []
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    print "Cleaning and parsing the training set movie reviews...\n"
    # first do the positives
    reviews = f_pos.readlines()
    for review in reviews:
        # pass in the review, clean, and get rid of stop words
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review, True)))
        labels.append(1)


    # clean the negative reviews
    reviews = f_neg.readlines()
    for review in reviews:
        # pass in the review, clean, and get rid of stop words
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review, True)))
        labels.append(0)

    # generate shuffle indices
    shuffle_indices = np.array(np.random.permutation(np.arange(len(clean_train_reviews))))

    # convert to numpy array so we can shuffle
    clean_train_reviews = np.array(clean_train_reviews)
    labels = np.array(labels)

    # shuffle the data
    clean_train_reviews = clean_train_reviews[shuffle_indices]
    labels = labels[shuffle_indices]

    # define train, test
    train_n = 100000

    # test data
    # d_test = clean_train_reviews[train_n:]
    # l_test = labels[train_n:]
    #
    # # train data
    # d_train = clean_train_reviews[0:train_n]
    # l_train = labels[0:train_n]


    d_test = []
    l_test = []
    d_train =[]
    l_train = []

    # extract training and test data
    d_test.extend(clean_train_reviews[train_n:train_n + 182])
    d_test.extend(clean_train_reviews[train_n + 182 + 100000:])

    l_test.extend(labels[train_n:train_n + 182])
    l_test.extend(labels[train_n + 182 + 100000:])

    d_train.extend(clean_train_reviews[0:train_n])
    d_train.extend(clean_train_reviews[train_n + 182:train_n + 182 + 100000])

    l_train.extend(labels[0:train_n])
    l_train.extend(labels[train_n + 182:train_n + 182 + 100000])


    # convert back to numpy array
    d_test = np.array(d_test)
    l_test = np.array(l_test)

    d_train = np.array(d_train)
    l_train = np.array(l_train)

    # d_test = np.vstack((clean_train_reviews[train_n:train_n + 182], clean_train_reviews[train_n + 182 + 100000:]))
    # l_test = np.vstack((labels[train_n:train_n + 182], labels[train_n + 182 + 100000:]))
    #
    # d_train = np.vstack((clean_train_reviews[0:train_n], clean_train_reviews[train_n + 182:train_n + 182 + 100000]))
    # l_train = np.vstack((labels[0:train_n], labels[train_n + 182:train_n + 182 + 100000]))






    # ****** Create a bag of words from the training set
    #
    print "Creating the bag of words...\n"


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(d_train)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    # ******* Train a random forest using the bag of words
    #
    print "Training the random forest (this may take a while)..."


    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable

    # This may take a few minutes to run
    forest = forest.fit( train_data_features, l_train )


    print "Evaluating model on test data..."

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(d_test)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print "Predicting test labels...\n"
    pred = forest.predict(test_data_features)

    print "Confusion Matrix"
    cm = confusion_matrix(l_test, pred)

    print cm
    print "Confusion Matrix"
    cm = confusion_matrix(l_test, pred)

    print "Normalized Confusion Matrix"
    show_confusion_matrix(l_test, pred)

