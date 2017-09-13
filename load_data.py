# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:04:38 2017

@author: workshop
"""

import pickle
import numpy as np # process vector/matrix
import math

import matplotlib.pyplot as plt # plot image
from skimage.util.montage import montage2d # plot image in batch
from skimage.color import rgb2gray

PIXEL_VALUE_RANGE = 255.
IS_INCLUDED_DISGUST = True

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
label_to_emotion = {i : emotions[i] for i in range(NUMB_EMOTIONS)}
emotion_to_label = {emotions[i] : i for i in range(NUMB_EMOTIONS)}

pickle_file = None
if (IS_INCLUDED_DISGUST == True):
    pickle_file = 'faces_small.pickle'
else:
    pickle_file = 'faces_big.pickle'

with open(pickle_file, 'rb') as f:
    dataset = pickle.load(f)

keywords = dataset.keys()
train_dataset = dataset['train_dataset']
train_labels = dataset['train_labels']
test_dataset = dataset['test_dataset']
test_labels = dataset['test_labels']


def plot_images(plotted_images, caption):
    
    plotted_images = (plotted_images - PIXEL_VALUE_RANGE / 2) / PIXEL_VALUE_RANGE
    plotted_images = [rgb2gray(i) for i in plotted_images]; 
        
    if (type(caption) == str):
        fig, ax1 = plt.subplots(1, 1, figsize = (8, 8))
        fig_montage = montage2d(np.array(plotted_images), rescale_intensity=True)
        ax1.imshow(fig_montage, cmap = 'gray', interpolation = 'none')
        ax1.set_title(caption)
        ax1.axis('off')
    else:
        fig = plt.figure()
        width_subplots = int(math.sqrt(len(plotted_images)))
        total_subplots = width_subplots * width_subplots
        fig.subplots_adjust(hspace=.5)
        for index in range(total_subplots):
            subfig = fig.add_subplot(width_subplots, width_subplots, index + 1)
            imgplot = plt.imshow(plotted_images[index], cmap='gray', 
                                 interpolation='None')
            subfig.set_title(caption[index])
            subfig.set_yticklabels([]), subfig.set_xticklabels([])
        plt.show()

plot_images(train_dataset[:100], [label_to_emotion[i] for i in train_labels[:100]])
