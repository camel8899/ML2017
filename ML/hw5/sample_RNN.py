import numpy as np
import string
import sys
import keras.backend as K 
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import matthews_corrcoef, f1_score

'''
train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
'''
train_path = './data/train_data.csv'
test_path = './data/test_data.csv'
output_path = './data/ans_new.csv'

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 200
nb_epoch = 200 
batch_size = 128


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r',encoding = 'utf8') as f:
    
        tags = []
        articles = []
        tags_list = []
        tags_corr = dict()
        ini = [(i,0) for i in range(1,9)]
        tags_count = dict(ini)
        f.readline()
        
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                tags_count[(len(tag))] += 1
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
                        tags_corr[t] = 0    
                    else:
                        tags_corr[t] += 1

                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list,tags_corr,tags_count)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.55
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis = -1)
    
    precision=tp/(K.sum(y_pred,axis = -1)+ K.epsilon())
    recall=tp/(K.sum(y_true, axis = -1) + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (Y_data,X_data,tag_list,tags_corr,tags_count) = read_data(train_path,True)
    (_, X_test, _, _, _) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))
    #print tag distribution
    '''  
    fig, ax = plt.subplots()
    
    rect = ax.bar(np.arange(38),tags_corr.values(),0.8,color = 'b')
    ax.set_ylabel('Number of tags')
    ax.set_title('Number of tags')
    ax.set_xticks( np.arange(38))
    ax.set_xticklabels(tags_corr.keys(), rotation = 'vertical')
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.38)
    plt.show()
	'''
	  
    fig, ax = plt.subplots()
    
    rect = ax.bar(np.arange(8),tags_count.values(),0.8,color = 'b')
    ax.set_xlabel('Number of the tags in a book')
    ax.set_ylabel('Number of books')
    ax.set_title('Books v.s. number of its tags')
    ax.set_xticks( np.arange(8))
    ax.set_xticklabels(tags_count.keys())
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()
	
    '''
    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = tokenizer.word_index

    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print ('Padding sequences.')
    train_sequences = pad_sequences(train_sequences)
    max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    ###
    train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    
    ### get mebedding matrix from glove
    print ('Get embedding dict from glove.')
    embedding_dict = get_embedding_dict('./data/glove.6B.%dd.txt'%embedding_dim)
    print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    ### build model
    print ('Building model.')

    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_article_length, trainable=False))
    model.add(GRU(256,activation='tanh',dropout=0.5, return_sequences = True,recurrent_dropout = 0.5))
    model.add(GRU(256,activation='tanh',dropout=0.5,recurrent_dropout = 0.5 ))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(38,activation='sigmoid'))
    
    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])
   
	checkpoint = ModelCheckpoint(filepath='best.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_f1_score',
                                 mode='max',
                                 period = 1)
    
    hist = model.fit(X_train, Y_train, 
                     validation_data=(X_val, Y_val),
                     epochs=nb_epoch, 
                     batch_size=batch_size,
                     callbacks=[checkpoint])
    
    model.save('hw5_train.h5')
   
    ### dynamic threshold from validation set
    out = model.predict(X_val)
    out = np.array(out)
    #print(f1_score(Y_val, out, average = 'micro'))
    
    test_pred = model.predict(test_sequences)
    thresh = 0.53
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        test_pred_thresh = np.zeros((1234, 38))
        for i in range(test_pred.shape[0]):
            for j in range(38):
                test_pred_thresh[i, j] = (test_pred [i, j] > thresh ).astype('int')
        for index,labels in enumerate(test_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    '''
if __name__=='__main__':
    main()
