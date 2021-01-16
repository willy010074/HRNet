# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:24:55 2019

@author: VPLab-2
"""
import cv2
import os
import math
import xlwt
#import pandas as pd

test_file_path = "sample/20200529-1/"
image_list = os.listdir(test_file_path)

mean_rmse = 0
pixel_num = 1244*376
index = 1

wb = xlwt.Workbook()
wb_sheet = wb.add_sheet('RMSE')

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
            total = total + (a - b)*(a - b)
    
    total = total/pixel_num
    total = math.sqrt(total)
    
    mean_rmse = mean_rmse + total
    wb_sheet.write(index, 0, gt_name)
    wb_sheet.write(index, 1, total)
    index += 1
#    print("RMSE : ", total)
#    print("----------------------------------------------------------------------")
#    
#print("mean RMSE : ", mean_rmse/2270)
wb.save('result_compare/RMSE.xls')