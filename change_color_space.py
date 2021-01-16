# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:24:55 2019

@author: VPLab-2
"""

import os
import cv2
import pickle
import matplotlib.pyplot as plt

plasma = plt.get_cmap('jet')

name = 'test/depth/'
img_list = os.listdir(name)

for i in img_list: 
    img = cv2.imread(name+i,0)
    img = img - img.min()
    img = img / img.max() * 255
    img = img.astype('uint8')
    #img = 255 - img
    #print (img.max(), img.min())
    colordepth = (plasma(img)[:,:,:3]*255).astype('uint8')
    
    cv2.imwrite("test/colordepth/20200529-1/" + i, colordepth)