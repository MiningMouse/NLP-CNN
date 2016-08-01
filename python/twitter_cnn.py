"""
Train convolutional network for sentiment analysis. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For 'CNN-non-static' gets to 82.1% after 61 epochs with following settings:
embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 3
dropout_prob = (0.7, 0.8)
hidden_dims = 100

For 'CNN-rand' gets to 78-79% after 7-8 epochs with following settings:
embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

For 'CNN-static' gets to 75.4% after 7 epochs with following settings:
embedding_dim = 100
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

* it turns out that such a small data set as "Movie reviews with one
sentence per review"  (Pang and Lee, 2005) requires much smaller network
than the one introduced in the original article:
- embedding dimension is only 20 (instead of 300; 'CNN-static' still requires ~100)
- 2 filter sizes (instead of 3)
- higher dropout probabilities and
- 3 filters per filter size is enough for 'CNN-non-static' (instead of 100)
- embedding initialization does not require prebuilt Google Word2Vec data.
Training Word2Vec on the same "Movie reviews" data set is enough to
achieve performance reported in the article (81.6%)

** Another distinct difference is slidind MaxPooling window of length=2
instead of MaxPooling over whole feature map as in the article
"""

import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
classes = ['0','1']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.jet):
    plt.figure()
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


def test_network(architecture, weights, data):
    model = model_from_json(open(architecture).read())
    model.load_weights(weights)
    pred = model.predict_classes(data, batch_size=1)
    # score = model.predict_proba(data, batch_size=1)
    return pred

training = True
if training == True:
    np.random.seed(2)

    # Parameters
    # ==================================================
    #
    # Model Variations. See Kim Yoon's Convolutional Neural Networks for
    # Sentence Classification, Section 3 for detail.

    model_variation = 'CNN-non-static' #'CNN-rand' #| CNN-non-static | CNN-static

    print('Model variation is %s' % model_variation)

    # Model Hyperparameters
    # embedding_dim = 20
    # filter_sizes = (3, 4)
    # num_filters = 150
    # dropout_prob = (0.25, 0.5)
    # hidden_dims = 150

    embedding_dim = 20
    filter_sizes = (3, 4)
    num_filters = 3
    dropout_prob = (0.7, 0.8)
    hidden_dims = 100

    # Training parameters
    batch_size = 32
    num_epochs = 7
    val_split = 0.1

    # Word2Vec parameters, see train_word2vec
    min_word_count = 1  # Minimum word count
    context = 10        # Context window size

    # Data Preparatopn
    # ==================================================
    #
    # Load data
    print("Loading data...")
    neg_train_path = './data/twitter_train_100000_and_test.neg'
    pos_train_path = './data/twitter_train_100000_and_test.pos'

    x, y, vocabulary, vocabulary_inv = data_helpers.load_data(pos_train_path,neg_train_path)

    if model_variation=='CNN-non-static' or model_variation=='CNN-static':
        embedding_weights = train_word2vec(x, vocabulary_inv, model_variation + 'twitter', embedding_dim, min_word_count, context)
        if model_variation=='CNN-static':
            try:
                x = embedding_weights[0][x]
            except MemoryError:
                print("Caught Memory Error, word list too large. Breaking data apart.")
                for i in xrange(len(embedding_weights[0])):
                    x.append(embedding_weights[0][x[i]])

    elif model_variation=='CNN-rand':
        embedding_weights = None
    else:
        raise ValueError('Unknown model variation')

    # define train, test and val sets. Validation is generated by model (keras splits data)
    train_n = 100000

    # pos = 182
    # neg = 177

    # get the test data at the end of the array
    # d_pos_test = x[train_n:train_n+182]
    # d_neg_test = x[train_n+182+100000:]
    #
    # l_pos_test = y[train_n:train_n + 182]
    # l_neg_test = y[train_n + 182 + 100000:]

    d_test = np.vstack((x[train_n:train_n+182], x[train_n+182+100000:]))
    l_test = np.vstack((y[train_n:train_n+182], y[train_n+182+100000:]))

    x = np.vstack((x[0:train_n], x[train_n+182:train_n+182+100000] ))
    y = np.vstack((y[0:train_n], y[train_n+182:train_n+182+100000] ))

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices].argmax(axis=1)

    print("Vocabulary Size: {:d}".format(len(vocabulary)))

    # Building model
    # ==================================================
    #
    # graph subnet with one input and one output,
    # convolutional layers concateneted in parallel

    # make sequence length variable depending on data formatting
    sequence_length = max(len(x) for x in x_shuffled)

    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    if not model_variation=='CNN-static':
        model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                            weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    # Training model
    # ==================================================
    model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
              nb_epoch=num_epochs, validation_split=val_split, verbose=2)

    # # save model architecture
    # json_string = model.to_json()
    # open('twitter_' + model_variation + '7_arch.json', 'w').write(json_string)

    # # save model weights
    # model_name = 'twitter_' + model_variation + str(num_epochs) + '.h5'
    # model.save_weights(model_name)

    # Run model against test data
    pred = model.predict_classes(d_test, batch_size=1)

    print "Confusion Matrix"
    cm = confusion_matrix(l_test.argmax(axis=1), pred)

    print "Normalized Confusion Matrix"
    show_confusion_matrix(l_test.argmax(axis=1), pred)
# else:
#
#     model_variation = 'CNN-rand'  # | CNN-non-static | CNN-static
#
#     # Model Hyperparameters
#     embedding_dim = 20
#     filter_sizes = (3, 4)
#     num_filters = 150
#     dropout_prob = (0.25, 0.5)
#     hidden_dims = 150
#
#     # Training parameters
#     batch_size = 32
#     num_epochs = 7
#     val_split = 0.1
#
#     # Word2Vec parameters, see train_word2vec
#     min_word_count = 1  # Minimum word count
#     context = 10  # Context window size
#
#     print("Loading data...")
#     neg_train_path = './data/twitter_test.neg'
#     pos_train_path = './data/twitter_test.pos'
#
#     x, y, vocabulary, vocabulary_inv = data_helpers.load_data(pos_train_path, neg_train_path)
#
#     if model_variation == 'CNN-non-static' or model_variation == 'CNN-static':
#         embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
#         if model_variation == 'CNN-static':
#             x = embedding_weights[0][x]
#     elif model_variation == 'CNN-rand':
#         embedding_weights = None
#     else:
#         raise ValueError('Unknown model variation')
#
#     # Shuffle data
#     test_num = 100
#     shuffle_indices = np.random.permutation(np.arange(len(y)))
#     x_shuffled = x[shuffle_indices]
#     y_shuffled = y[shuffle_indices].argmax(axis=1)
#     pred = test_network('twitter_' + model_variation + '7_arch.json', 'twitter_' + model_variation + '7.h5',x_shuffled[0:test_num])
#
#     l = y_shuffled[0:test_num]
#
#     confusion_matrix(l, pred)
#
#     show_confusion_matrix(l, pred)