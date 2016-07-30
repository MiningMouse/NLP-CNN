'''
Author: Austin Dress
Purpose: This file is used to clean up the twitter_cnn data and prepare it for training and testing.
The twitter data was already cleaned for BOW classifier and saved as a pickle. This script reads this data in
and saves it into csv files for train/test positive/negative sentiment.
Data can be found here: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
Date: 7/19/16
'''


import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from imdb.KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk
import pickle


if __name__ == '__main__':

    # read in the raw train and test data
    print "Loading the data"

    # Load the twitter data pickle. Data was already cleaned for bow, need to convert to format for

    with open('data/sent_labels_cleaned2.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        d_train, l_train, d_test, l_test = pickle.load(f)


    # get the number of exemplars

    # num_exemplars = len(data["review"])
    # train_n = int(num_exemplars*0.8)
    #
    # # shuffle the data
    # rand = np.random.RandomState(0)
    # shuffle = rand.permutation(num_exemplars)
    # data = data.iloc[np.random.permutation(num_exemplars)]
    #
    #
    # # train = data[0:train_n]
    # # test = data[train_n:]
    # print ('Train Size: ' + str(train_n) +  ' Test Size: ' + str(num_exemplars-train_n) )

    # create files to save positive and negative training examples
    fp = open('twitter_train.pos', 'w')
    fn = open('twitter_train.neg','w')
    count_pos = 0
    count_neg = 0
    for i in xrange(0, len(l_train)):
        if  l_train[i] == 4:
            fp.write(d_train[i])
            fp.write('\n')
            count_pos += 1
        else:
            fn.write(d_train[i])
            fn.write('\n')
            count_neg += 1
    fp.close()
    fn.close()
    print "Train Negative and Positive Data Count:"
    print count_neg, count_pos


    count_pos = 0
    count_neg = 0
    count_neu = 0
    # create files to save positive and negative test examples
    fp = open('twitter_test.pos','w') # positve
    fn = open('twitter_test.neg','w') # negative
    fneu = open('twitter_test.neg', 'w') # neutral

    for i in xrange(0, len(l_test)):
        if l_test[i] == 4:
            fp.write(d_test[i])
            fp.write('\n')
            count_pos +=1
        elif l_test[i] == 2:
            fneu.write(d_test[i])
            fneu.write('\n')
            count_neu += 1
        else:
            fn.write(d_test[i])
            fn.write('\n')
            count_neg += 1

    fp.close()
    fn.close()
    fneu.close()

    print "Test Negative, Positive, Neutral Data Count:"
    print count_neg, count_pos, count_neu