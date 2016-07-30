#!/usr/bin/env python

#  Author: Austin Dress
#  Date: 6/26/16
#
#  This file is for training and testing on the twitter
#  sentiment140 dataset using bow of words(BOW)
#
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk
import pickle

import re

# compile regular expressions that match repeated characters and emoji unicode
emoji = re.compile(u'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]',re.UNICODE)
multiple = re.compile(r"(.)\1{1,}", re.DOTALL)

def format(tweet):

    #strip emoji
    stripped = emoji.sub('',tweet)

    #strip URLs
    stripped = re.sub(r'http[s]?[^\s]+','', stripped)

    #strip "@name" components
    stripped = re.sub(r'(@[A-Za-z0-9\_]+)' , "" ,stripped)

    #strip html '&amp;', '&lt;', etc.
    stripped = re.sub(r'[\&].*;','',stripped)

    #strip punctuation
    stripped = re.sub(r'[#|\!|\-|\+|:|//]', " ", stripped)

    #strip the common "RT"
    stripped = re.sub( 'RT.','', stripped)

    #strip whitespace down to one.
    stripped = re.sub('[\s]+' ,' ', stripped).strip()

    #strip multiple occurrences of letters
    stripped = multiple.sub(r"\1\1", stripped)

    #strip all non-latin characters
    #if we wish to deal with foreign language tweets, we would need to first
    #translate them before taking this step.

    stripped = re.sub('[^a-zA-Z0-9|\']', " ", stripped).strip()

    return stripped

data_path = '/home/dressag1/Documents/Datasets/imdb_movie/'
if __name__ == '__main__':

    print "Loading the data"
    # Getting back the objects:
    with open('sent_labels_cleaned2.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        d_train, l_train, d_test, l_test = pickle.load(f)

    # print "Combining words into sentences for BOW"
    # d_train_sen = []
    # for d in d_train:
    #     tweet = format(' '.join(d))
    #     d_train_sen.append(tweet)
    #
    # d_test_sen = []
    # for d in d_test:
    #     tweet = format(' '.join(d))
    #     d_test_sen.append(tweet)


    # train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', data_path +'labeledTrainData.tsv'), header=0, \
    #                 delimiter="\t", quoting=3)
    # test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', data_path + 'testData.tsv'), header=0, delimiter="\t", \
    #                quoting=3 )
    #
    # print 'The first review is:'
    # print train["review"][0]
    #
    # raw_input("Press Enter to continue...")
    #
    #
    # print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...'
    # #nltk.download()  # Download text data sets, including stop words
    #
    # # Initialize an empty list to hold the clean reviews
    # clean_train_reviews = []
    #
    # # Loop over each review; create an index i that goes from 0 to the length
    # # of the movie review list
    #
    # print "Cleaning and parsing the training set movie reviews...\n"
    # for i in xrange( 0, len(train["review"])):
    #     clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
    #
    #
    # ****** Create a bag of words from the training set

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
    train_data_features = (train_data_features).toarray # changed from to_array() Memory Error

    # ******* Train a random forest using the bag of words
    #
    print "Training the random forest (this may take a while)..."


    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, l_train )
    #
    #
    #
    # # Create an empty list and append the clean reviews one by one
    # clean_test_reviews = []
    #
    # print "Cleaning and parsing the test set movie reviews...\n"
    # for i in xrange(0,len(test["review"])):
    #     clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))
    #
    # # Get a bag of words for the test set, and convert to a numpy array
    # test_data_features = vectorizer.transform(clean_test_reviews)
    # test_data_features = test_data_features.toarray()
    #
    # # Use the random forest to make sentiment label predictions
    # print "Predicting test labels...\n"
    # result = forest.predict(test_data_features)
    #
    # # Copy the results to a pandas dataframe with an "id" column and
    # # a "sentiment" column
    # output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    #
    # # Use pandas to write the comma-separated output file
    # output.to_csv(os.path.join(os.path.dirname(__file__), 'data', data_path + 'Bag_of_Words_model.csv'), index=False, quoting=3)
    # print "Wrote results to Bag_of_Words_model.csv"
    #
    #
