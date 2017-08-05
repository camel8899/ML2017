import os
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model
from keras import backend as K
from utils import *
import numpy as np
NUM_STEPS = 20
RECORD_FREQ = 5

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
img_dir = os.path.join(base_dir, 'image')
filter_dir = os.path.join(img_dir,'filter')
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    global filter_images
    lr = 1
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data])
        input_image_data += grads_value * lr
        if(num_step%RECORD_FREQ == 0):
            filter_images.append((img, loss_value))
        if loss_value <= 0.:
            break
    return filter_images

def main():
    parser = argparse.ArgumentParser(prog='gradient_ascent.py',
            description='ML-Assignment3 gradient ascent.')
    parser.add_argument('--model',type=str,default='shallow',choices=['shallow'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=80)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    model_path = os.path.join(exp_dir,store_path)

    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input

    name_ls = ["Conv2D_1"]
    collect_layers = [ layer_dict[name].output for name in name_ls ]
    nb_filter = 32
    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])

            ###
            "You need to implement it."
            filter_imgs = grad_ascent(NUM_STEPS, input_img_data, iterate)
            ###

        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16, 16, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            img_path = os.path.join(filter_dir, '{}-{}'.format(store_path, name_ls[cnt]))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

if __name__ == "__main__":
    main()