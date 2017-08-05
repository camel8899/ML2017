import os
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model
from keras import backend as K
from utils import *
from marcos import *
import numpy as np
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
img_dir = os.path.join(base_dir, 'image')
filter_dir = os.path.join(img_dir,'filter')

def main():
    parser = argparse.ArgumentParser(prog='output_filter.py',
            description='ML-Assignment3 output_filter.')
    parser.add_argument('--model',type=str,default='shallow',choices=['shallow'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=80)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    model_path = os.path.join(exp_dir,store_path,'model.hdf5')

    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[:])

    input_img = emotion_classifier.input
    name_ls = ["conv2d_1"]
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    figures = read_dataset('train',4000)
    figures = [ np.reshape(figures[i],(1, 48, 48, 1)) 
                       for i in range(len(figures)) ]
    choose_id = 17
    photo = figures[choose_id]
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        img_path = os.path.join(filter_dir, store_path)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))