# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:24:55 2019

@author: VPLab-2
"""

import cv2
import os
import xlwt
#import pandas as pd

test_file_path = "sample/20200520-1/"
kitti_ori_gt_path = "KITTI_depth/"
image_list = os.listdir(test_file_path)
total_thr1 = 0
total_thr2 = 0
total_thr3 = 0
index = 1

wb = xlwt.Workbook()
wb_sheet = wb.add_sheet('threshold')

for name in image_list:
    print(name + ":" + "threshold")
    test = cv2.imread(test_file_path + name,0)
    gt_name = name.replace("result_","")
    gt = cv2.imread("kitti_dataset/resize_depth/" + gt_name,0)
    ori_gt = cv2.imread(kitti_ori_gt_path + gt_name,0)
    ori_gt = cv2.resize(ori_gt, (1244, 376), interpolation=cv2.INTER_NEAREST)
    #gt = cv2.imread("nyu_dataset/resize_depth/" + gt_name,0)
        
    total = 0
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    d_max = 0
    thershold = 1.25
    
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if int(ori_gt[i,j]) != 0:
                total += 1
                a = int(test[i,j])
                b = int(gt[i,j])
                d = max(a/b, b/a)
                
                if d > d_max:
                    d_max = d
                    
                if d < thershold:
                    correct_1 += 1
                
                if d < thershold*thershold:
                    correct_2 += 1
                
                if d < thershold*thershold*thershold:
                    correct_3 += 1
    
    total_thr1 += correct_1/total
    total_thr2 += correct_2/total
    total_thr3 += correct_3/total
    
    wb_sheet.write(index, 0, gt_name)
    wb_sheet.write(index, 1, correct_1/total)
    wb_sheet.write(index, 2, correct_2/total)
    wb_sheet.write(index, 3, correct_3/total)
    index += 1
    
#    print("threshold 1 : ", correct_1/total)
#    print("threshold 2 : ", correct_2/total)
#    print("threshold 3 : ", correct_3/total)
#    print("---------------------")
#

wb.save('result_compare/threshold.xls')

#print("mean threshold 1 : ", total_thr1/2270)
#print("mean threshold 2 : ", total_thr2/2270)
#print("mean threshold 3 : ", total_thr3/2270)
#print("---------------------")
#test = cv2.imread('result3.png',0)
#gt = cv2.imread('gt.png',0)
#
##test = test-test.min()
##gt = gt-gt.min()
##
##test_max = test.max()
##gt_max = gt.max()
##
##for i in range(test.shape[0]):
##    for j in range(test.shape[1]):
##        test[i,j] = int(test[i,j]*255/test_max)
##        gt[i,j] = int(gt[i,j]*255/gt_max)
#
#total = test.shape[0]*test.shape[1]
#correct_1 = 0
#correct_2 = 0
#correct_3 = 0
#d_max = 0
#thershold = 1.25
#
#for i in range(test.shape[0]):
#    for j in range(test.shape[1]):
#        d = max(test[i,j]/gt[i,j], gt[i,j]/test[i,j])
#        
#        if d > d_max:
#            d_max = d
#            
#        if d < thershold:
#            correct_1 += 1
#        
#        if d < thershold*thershold:
#            correct_2 += 1
#        
#        if d < thershold*thershold*thershold:
#            correct_3 += 1
#
#print("threshold 1 : ", correct_1/total)
#print("threshold 2 : ", correct_2/total)
#print("threshold 3 : ", correct_3/total)
        