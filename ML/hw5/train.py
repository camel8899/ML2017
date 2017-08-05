import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adamax, SGD, Adam, Adadelta
from keras import backend as K
#from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)



if __name__ == '__main__':
    ###### Parsing by regular expression ######
    f = open('./data/train_data.csv','r')
    line = f.readlines()
    tags = [re.findall('\"(.+?)\"', line[i])[0] for i in range(1, len(line))]
    texts = [re.sub('\d+,\"(.+?)\",', '', line[i]) for i in range(1, len(line))]
    
    ####### Preprocessing Texts ############
    #tokenizer (convert word to sequences to word index sequences)
    tokenizer_texts = Tokenizer(split = ' ')
    tokenizer_texts.fit_on_texts(texts)
    text_sequences = tokenizer_texts.texts_to_sequences(texts)
    text_index = tokenizer_texts.word_index
    print('Found %s unique tokens in texts' % len(text_index))
    #print(len(text_sequences[100]))
    #padding sequences to equal length
    data = pad_sequences(text_sequences)
    data = np.array(data)
    print('Shape of data tensor: ', data.shape)
    #data = data.reshape((4964, 306, 1))

    ######## Preprocessing Tags ############
    '''
    tokenizer_tags = Tokenizer(split = ' ')
    tokenizer_tags.fit_on_texts(tags)
    tag_sequences = tokenizer_tags.texts_to_sequences(tags)
    tag_index = tokenizer_tags.word_index
    print('Found %s unique tokens in tags' % len(tag_index)) 
    print(tag_index)
    '''
    tag_index = []  
    for i ,obj in enumerate(tags):
        row = obj.split(' ')
        for j, col in enumerate(row):
            flag = 0
            for k , comp in enumerate(tag_index):
                if col == comp:
                    flag = 1
            if flag == 0:
                tag_index.append(col)
    #print(len(tag_index))
    tag = []
    for i ,obj in enumerate(tags):
        row = obj.split(' ')
        tag_row = []
        for j, col in enumerate(row):
            for k, comp in enumerate(tag_index):
                if comp == col:
                    tag_row.append(k)
        tag.append(tag_row)
        #print(str(i) + "-th row:" + str(tag[i]))
    tag = MultiLabelBinarizer().fit_transform(tag)
    #print(tag.shape)

    
    ########## Layers #########
    
    print('Start Building Model...')
    
    model = Sequential()
    model.add(Embedding(len(text_index)+1, 1024, input_length = 306))

    model.add(LSTM(128, activation = 'relu', dropout = 0.2, recurrent_dropout = 0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(38, activation = 'sigmoid'))
    '''
    embedding_layer = Embedding(len(text_index), 64, input_length = 306, trainable = False)
    '''
    ########## Compilation ###########
    adamax = Adamax(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = 'binary_crossentropy', metrics = [fmeasure,precision,recall], optimizer = 'SGD')
    model.summary()
    model.save('hw5_train.h5')
    model.fit(data, tag, batch_size = 128,epochs = 15, validation_split = 0.2)

    