import argparse
import os
import model
import random as rand
from utils import * 
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

base_dir = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(base_dir,'exp')

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

def main():
    aug = True

    parser = argparse.ArgumentParser(prog='hw3_integrate.py',
            description='ML-Assignment training script.')
    parser.add_argument('--model',type=str,default='shallow',choices=['shallow'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=80)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    args = parser.parse_args()
    # set the path
    dir_cnt = 0
    log_path = "{}_epoch{}".format(args.model,str(args.epoch))
    log_path += '_'
    store_path = os.path.join(exp_dir,log_path+str(dir_cnt))
    while dir_cnt < 30:
        if not os.path.isdir(store_path):
            os.mkdir(store_path)
            break
        else:
            dir_cnt += 1
            store_path = os.path.join(exp_dir,log_path+str(dir_cnt))

    emotion_classifier = model.build_model(args.model)
    tr_feats,tr_labels,_ = read_dataset('train')

    choose = rand.sample(range(0,tr_feats.shape[0]-1),4000)
    dev_feats = tr_feats[choose]
    dev_labels = tr_labels[choose]
    tr_feats = np.delete(tr_feats,choose,axis = 0)
    tr_labels = np.delete(tr_labels,choose,axis = 0)

    history = History()

    if aug is False:
        emotion_classifier.fit(x = tr_feats,y = tr_labels,
            batch_size=args.batch,nb_epoch=args.epoch,validation_data=(dev_feats,dev_labels),
            callbacks=[history])
    else:
        datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        zca_whitening = False,
        rotation_range=3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip = False)
        datagen.fit(tr_feats)
        emotion_classifier.fit_generator(datagen.flow(tr_feats,tr_labels,batch_size = args.batch),
        steps_per_epoch = int(tr_feats.shape[0]/args.batch),epochs = args.epoch,
        validation_data = (dev_feats,dev_labels),callbacks = [history])

    dump_history(store_path,history)
    emotion_classifier.save(os.path.join(store_path,'model.h5'))

if __name__ == "__main__":
    main()

