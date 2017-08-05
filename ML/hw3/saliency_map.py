import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.realpath(__file__))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
exp_dir = os.path.join(base_dir, 'exp')

def main():
    parser = argparse.ArgumentParser(prog='saliency_map.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--model',type=str,default='shallow',choices=['shallow'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=80)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    model_name = "model.h5"
    model_path = os.path.join(exp_dir,store_path,model_name)
    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_path), 'yellow', attrs=['bold']))

    private_pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
    private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
                       for i in range(len(private_pixels)) ]
    input_img = emotion_classifier.input
    img_ids = ["image ids from which you want to make heatmaps"]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = None
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''

        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, 'privateTest', '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, 'privateTest', '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()