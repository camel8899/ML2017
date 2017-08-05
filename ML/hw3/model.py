import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation
from keras.layers import Conv2D ,MaxPooling2D
from keras.optimizers import SGD,Adamax
nb_class = 7

def build_model(mode):
    model = Sequential()
    if mode == 'shallow':
        model = Sequential()
        model.add(Conv2D(64,(3,3),input_shape=(48,48,1),activation = 'relu'))
        model.add(Conv2D(64,(3,3),activation = 'relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128,(3,3),activation = 'relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256,(3,3),activation = 'relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(units=1500,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=7,activation='softmax'))
      

        opt = Adamax(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
        model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary() 
    return model
