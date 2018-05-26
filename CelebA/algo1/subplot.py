import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from os import path
import math
from scipy.misc import imread

def plot_images():
    folder = path.realpath("i1")
    images = os.listdir(folder)
    print(images)
    i = 0
    #plt.imshow(images[0])
    plt.savefig('./' + "here" + '.jpg')
    plt.figure(figsize=[30,15]) # set image size
    plt.subplots_adjust(wspace = 0)# set distance between the subplots
    for image in images:
        plt.subplot(1,len(images),i+1)
        im = imread(folder+'/'+image)
        i+=1
        imgplot = plt.imshow(im) 
        imgplot.axes.get_xaxis().set_visible(False)   
        imgplot.axes.get_yaxis().set_visible(False)    
        plt.axis('off')
        plt.savefig('./' + "here" + '.jpg')
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.show()
    return 

plot_images()