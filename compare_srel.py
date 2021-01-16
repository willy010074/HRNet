# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:24:55 2019

@author: VPLab-2
"""
import cv2
import os
import xlwt
#import pandas as pd

test_file_path = "sample/20200529-1/"
image_list = os.listdir(test_file_path)

mean_rel = 0
pixel_num = 1244*376
index = 1

wb = xlwt.Workbook()
wb_sheet = wb.add_sheet('srel')

for name in image_list:
    print(name + ":")
    test = cv2.imread(test_file_path + name,0)
    gt_name = name.replace("result_","")
    gt = cv2.imread("kitti_dataset/resize_depth/" + gt_name,0)
    #gt = cv2.imread("nyu_dataset/resize_depth/" + gt_name,0)
    
    total = 0
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            a = int(test[i,j])
            b = int(gt[i,j])
            c = a - b
            temp = (abs(c)*abs(c))/gt[i,j]
            total = total + temp #16/219
    
    total = total/pixel_num
    mean_rel = mean_rel + total
    
    wb_sheet.write(index, 0, gt_name)
    wb_sheet.write(index, 1, total)
    index += 1
    
#    print("srel : ", total)
#    print("----------------------------------------------------------------------")
#    
#print("mean srel : ", mean_rel/2270)
wb.save('result_compare/srel.xls')