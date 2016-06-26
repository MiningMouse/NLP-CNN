import pandas as pd
import os
import re
from KaggleWord2VecUtility import KaggleWord2VecUtility
import nltk
import pickle
import logging
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

data_path = '/home/dressag1/Documents/Datasets/twitter_sentiment/trainingandtestdata/'

#compile regular expressions that match repeated characters and emoji unicode
# emoji = re.compile(u'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]',re.UNICODE)
# multiple = re.compile(r"(.)\1{1,}", re.DOTALL)
#
# def format(tweet):
#
#     #strip emoji
#     stripped = emoji.sub('',tweet)
#
#     #strip URLs
#     stripped = re.sub(r'http[s]?[^\s]+','', stripped)
#
#     #strip "@name" components
#     stripped = re.sub(r'(@[A-Za-z0-9\_]+)' , "" ,stripped)
#
#     #strip html '&amp;', '&lt;', etc.
#     stripped = re.sub(r'[\&].*;','',stripped)
#
#     #strip punctuation
#     stripped = re.sub(r'[#|\!|\-|\+|:|//]', " ", stripped)
#
#     #strip the common "RT"
#     stripped = re.sub( 'RT.','', stripped)
#
#     #strip whitespace down to one.
#     stripped = re.sub('[\s]+' ,' ', stripped).strip()
#
#     #strip multiple occurrences of letters
#     stripped = multiple.sub(r"\1\1", stripped)
#
#     #strip all non-latin characters
#     #if we wish to deal with foreign language tweets, we would need to first
#     #translate them before taking this step.
#
#     stripped = re.sub('[^a-zA-Z0-9|\']', " ", stripped).strip()
#
#     return stripped


do_data_prep = False
remove_stop_words = False

if __name__ == '__main__':


    if do_data_prep:

        name_list = ['sentiment', 'id', 'time', 'query', 'user', 'text']
        train = pd.read_csv(data_path + "training.1600000.processed.noemoticon.csv", \
                         header=None, names=name_list)

        test = pd.read_csv(data_path + "testdata.manual.2009.06.14.csv", \
                         header=None, names=name_list)

        # Initialize an empty list to hold the clean tweets
        clean_train_tweets = []

        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list
        print "Cleaning and parsing the training set tweets...\n"
        for i in xrange(0, len(train["text"])):
            print (str(i))
            clean_train_tweets.append(KaggleWord2VecUtility.review_to_wordlist(train["text"][i], remove_stop_words))

        train_labels = train['sentiment'].tolist()

        # Initialize an empty list to hold the clean tweets
        clean_test_tweets = []

        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list
        print "Cleaning and parsing the test set tweets...\n"
        for i in xrange(0, len(test["text"])):
            print (str(i))
            clean_test_tweets.append(KaggleWord2VecUtility.review_to_wordlist(test["text"][i], remove_stop_words))

        test_labels = test['sentiment'].tolist()

        print "Saving the data"
        # Saving the objects:
        with open('sent_labels_cleaned.pickle', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump([clean_train_tweets, train_labels, clean_test_tweets, test_labels], f)

    print "Loading the data"
    # Getting back the objects:
    with open('sent_labels_cleaned.pickle') as f:  # Python 3: open(..., 'rb')
        d_train, l_train, d_test, l_test = pickle.load(f)


    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 8       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model = Word2Vec(d_train, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context_twitter140"
    model.save(model_name)