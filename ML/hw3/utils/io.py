import os
import numpy as np
import random
from keras.utils.np_utils import to_categorical

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(base_dir,'data')


def read_dataset(mode='train',isFeat=True):
    """
    Return:
        # features: (int. list) list
        # labels: int32 2D array
        data_ids: int. list
    """
    # num_data = 0
    datas = []

    with open(os.path.join(data_dir,'{}.csv'.format(mode))) as file:
        for line_id,line in enumerate(file):
            if line_id == 0:
                continue
            if isFeat:
                label, feat=line.split(',')
            else:
                _,feat = line.split(',')
            feat = np.fromstring(feat,dtype=int,sep=' ')
            # print(feat)
            feat = np.reshape(feat,(48,48,1))

            if isFeat:
                datas.append((feat,int(label),line_id))
            else:
                datas.append(feat)

    # random.shuffle(datas)  # shuffle outside
    if isFeat:
        feats,labels,line_ids = zip(*datas)
    else:
        feats = datas
    feats = np.asarray(feats)
    if isFeat:
        labels = to_categorical(np.asarray(labels,dtype=np.int32))
        return feats,labels,line_ids
    else:
        return feats


def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))