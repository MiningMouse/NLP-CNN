'''
Author: Austin Dress
Purpose: This file is used to clean up the imdb_cnn data and prepare it for training and testing
Date: 7/19/16
'''

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from imdb.KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk


data_path = '/home/dressag1/Documents/Datasets/imdb_movie/'
if __name__ == '__main__':

    # read in the raw train and test data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', data_path +'labeledTrainData.tsv'), header=0, \
                    delimiter="\t", quoting=3)

    # get the number of exemplars

    num_exemplars = len(data["review"])
    train_n = int(num_exemplars*0.8)

    # shuffle the data
    rand = np.random.RandomState(0)
    shuffle = rand.permutation(num_exemplars)
    data = data.iloc[np.random.permutation(num_exemplars)]


    # train = data[0:train_n]
    # test = data[train_n:]
    print ('Train Size: ' + str(train_n) +  ' Test Size: ' + str(num_exemplars-train_n) )

    # create files to save positive and negative training examples
    fp = open('imdb_train.pos', 'w')
    fn = open('imdb_train.neg','w')

    for i in xrange(0, train_n):
        if data['sentiment'][i] == 1:
            fp.write(data['review'][i])
            fp.write('\n')
        else:
            fn.write(data['review'][i])
            fn.write('\n')

    fp.close()
    fn.close()

    # create files to save positive and negative test examples
    fp = open('imdb_test.pos','w')
    fn = open('imdb_test.neg','w')

    for i in xrange(train_n, len(data)):
        if data['sentiment'][i] == 1:
            fp.write(data['review'][i])
            fp.write('\n')
        else:
            fn.write(data['review'][i])
            fn.write('\n')

    fp.close()
    fn.close()
