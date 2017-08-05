import numpy as np
import sys
import pandas as pd
import os
from keras.models import Sequential,Model 
from keras.layers import Input,Embedding, Reshape, Merge, Dropout,Flatten,Dot,Add 
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

embedding_dim = 16
split_ratio = 0.1

def split_data(U,M,R,split_ratio):
    indices = np.arange(U.shape[0])  
    np.random.shuffle(indices) 
    
    U_data = U[indices]
    M_data = M[indices]
    R_data = R[indices]
    
    num_validation_sample = int(split_ratio * U_data.shape[0] )
    
    U_train = U_data[num_validation_sample:]
    M_train = M_data[num_validation_sample:]
    R_train = R_data[num_validation_sample:]

    U_val = U_data[:num_validation_sample]
    M_val = M_data[:num_validation_sample]
    R_val = R_data[:num_validation_sample]

    return (U_train,M_train,R_train),(U_val,M_val,R_val)

def MF_model(n_users,n_movies,max_userid,max_movieid,embedding_dim = 10):
    
    User_input = Input(shape = [1])
    Movie_input = Input(shape = [1])
    User_embed = Embedding(output_dim = embedding_dim,input_dim = max_userid+1,embeddings_initializer = 'random_normal')(User_input)
    User_reshape = Flatten()(User_embed)
    Movie_embed = Embedding(output_dim = embedding_dim,input_dim = max_movieid+1,embeddings_initializer = 'random_normal')(Movie_input)
    Movie_reshape = Flatten()(Movie_embed)
    User_bias = Flatten()(Embedding(output_dim = 1,input_dim = max_userid+1,embeddings_initializer = 'zeros')(User_input))
    Movie_bias = Flatten()(Embedding(output_dim = 1,input_dim = max_userid+1,embeddings_initializer = 'zeros')(Movie_input))

    Main_dot = Dot(axes = 1)([User_reshape,Movie_reshape])
    Main_add = Add()([Main_dot,User_bias,Movie_bias])
    
    model = Model([User_input,Movie_input],Main_add)
    model.summary()
    model.compile(loss = 'mse',optimizer = 'adam')
    
    return model


def main():
    '''
    train_path = os.path.join(sys.argv[1],'train.csv')
    train_data = pd.read_csv(train_path)
    #users,movies,rating = read_data(train_path,True)

    users = np.array(train_data['UserID']).astype('int')
    movies = np.array(train_data['MovieID']).astype('int')
    rating = np.array(train_data['Rating']).astype('float')

    mean = rating.mean()
    std = rating.std()
    rating = (rating-mean)/std
    print(mean)
    print(std)
    max_userid = np.max(users)
    max_movieid = np.max(movies)

    print("Find maximum user id {} and maximum movie id {} in training set...".format(max_userid,max_movieid))

    checkpoint = ModelCheckpoint(filepath = 'best_16_add2.hdf5',verbose = 1,
            save_best_only = True, save_weights_only = True,monitor = 'val_loss',mode = 'auto')

    (U_train,M_train,R_train),(U_val,M_val,R_val) = split_data(users,movies,rating,split_ratio)
    model = MF_model(len(users),len(movies),max_userid, max_movieid,embedding_dim)
    model.fit([U_train,M_train],R_train,epochs = 12,batch_size = 32,validation_data = ([U_val,M_val],R_val),callbacks = [checkpoint])
    model.save('adam_new.h5')
    '''
    test_path = os.path.join(sys.argv[1],'test.csv')
    test_data = pd.read_csv(test_path)
    model = load_model('tk.h5')
    users = np.array(test_data['UserID']).astype('int')
    movies = np.array(test_data['MovieID']).astype('int')
    output = model.predict([users,movies])
    with open(sys.argv[2],'w') as f:
        f.write("TestDataID,Rating\n")
        for i,rating in enumerate(output):
            if rating[0] > 5:
                rating[0] = 5
            f.write("{},{}\n".format(i+1,rating[0]))

if __name__ == "__main__":
    main()
   
