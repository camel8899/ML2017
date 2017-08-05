import os
from termcolor import colored, cprint
import argparse
from keras.utils.vis_utils import plot_model
from keras.models import load_model

base_dir = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(base_dir,'exp')

def main():
    parser = argparse.ArgumentParser(prog='hw3_plot_model.py',
            description='Plot the model.')
    parser.add_argument('--model',type=str,default='shallow',choices=['shallow'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=80)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    model_path = os.path.join(exp_dir,store_path,'model.h5')

    emotion_classifier = load_model(model_path)
    emotion_classifier.summary()
    plot_model(emotion_classifier,to_file='model.png')

if __name__ == "__main__":
    main()
