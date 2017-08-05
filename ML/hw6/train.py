import numpy as np
import sys
from argparse import ArgumentParser
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import Callback


train_path = './data/train.csv'

embedding_dim = 10
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


def build_model(num_users, num_movies, lat_dims, mode):
    u_emb_input = Input(shape=(1,))
    u_emb = Embedding(num_users, lat_dims,
                      embeddings_initializer='random_normal')(u_emb_input)
    u_emb = Flatten()(u_emb)
    u_bias = Embedding(num_users, 1,
                       embeddings_initializer='zeros')(u_emb_input)
    u_bias = Flatten()(u_bias)
    m_emb_input = Input(shape=(1, ))
    m_emb = Embedding(num_movies, lat_dims,
                      embeddings_initializer='random_normal')(m_emb_input)
    m_emb = Flatten()(m_emb)
    m_bias = Embedding(num_movies, 1,
                       embeddings_initializer='zeros')(m_emb_input)
    m_bias = Flatten()(m_bias)
    if mode == 'mf':
        dot = Dot(axes=1)([u_emb, m_emb])
        output = Add()([dot, u_bias, m_bias])
        model = Model(inputs=[u_emb_input, m_emb_input], outputs=output)
    elif mode == 'dnn':
        u_emb = Dropout(DROPOUT_RATE)(u_emb)
        m_emb = Dropout(DROPOUT_RATE)(m_emb)
        concat = Concatenate()([u_emb, m_emb, u_info_emb, m_info_emb])
        dnn = Dense(256, activation='relu')(concat)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        dnn = BatchNormalization()(dnn)
        dnn = Dense(256, activation='relu')(dnn)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        dnn = BatchNormalization()(dnn)
        dnn = Dense(256, activation='relu')(dnn)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        dnn = BatchNormalization()(dnn)
        output = Dense(1, activation='relu')(dnn)
        model = Model(inputs=[u_emb_input, m_emb_input], outputs=output)

    return model


def main():
    """ Main function """
    parser = ArgumentParser()
    #parser.add_argument('--dnn', action='store_true', help='Use DNN model')
    parser.add_argument('--normal', action='store_true', help='Normalize ratings')
    parser.add_argument('--dim', type=int, default=128, help='Specify latent dimensions')
    args = parser.parse_args()
    train = pd.read_csv('./data/train.csv')
    max_userid = train['UserID'].drop_duplicates().max()
    max_movieid = train['MovieID'].drop_duplicates().max()
    
    users = train['UserID'].values - 1
    movies = train['MovieID'].values - 1
    ratings = train['Rating'].values
    
    if args.normal:
    	mean = np.mean(ratings)
    	std = np.std(ratings)
    	ratings = (ratings - np.mean(ratings)) / np.std(ratings)
    	
    model = build_model(max_userid, max_movieid, args.dim, 'mf')
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    checkpoint = ModelCheckpoint(filepath='best.hdf5',verbose=1,save_best_only=True,
	                                 save_weights_only=True,
	                                 monitor='val_loss',
	                                 mode='min')
    (U_train,M_train,R_train),(U_val,M_val,R_val) = split_data(users,movies,
		ratings,split_ratio)
    
    model.fit([U_train,M_train],R_train,epochs = 7,
		validation_data = ([U_val,M_val],R_val),callbacks = [checkpoint],batch_size = 256)
    model.save('my_mf_256.h5')
    
    test_path = './data/test.csv'
    test = pd.read_csv(test_path)
    output = model.predict([test['UserID'].values-1,test['MovieID'].values-1])
    if args.normal:
    	output = output*std+mean
    with open('ans_my_256.csv','w') as f:
    	f.write("TestDataID,Rating\n")
    	for i,rating in enumerate(output):
    		r = rating[0]
    		if r > 5:
    			r = 5
    		if r < 1:
    			r = 1
    		f.write("{},{}\n".format(i+1,r))


if __name__ == "__main__":
	main()

