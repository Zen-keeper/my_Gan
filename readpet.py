# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:55:55 2019

@author: jsc
"""
import os

import nibabel as nib
import numpy as np
import imageio   #转换成图像
import cv2
import matplotlib.pyplot as plt

img_list = []
label_list = []
dir_counter = 0
i=0

def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False
   

def read_img(pp,num):
    x_list = []
    y_list = []
    path = pp
    n = 0
    for child_dir in os.listdir(path):
        #child_path = path+"/"+child_dir
        child_path = os.path.join(path, child_dir)
        files = os.listdir(child_path)
        files.sort()
        for dir_image in files:
    
            if (endwith(dir_image,'.img')or endwith(dir_image,'.nii'))and n<num:
                #img = cv2.imread(os.path.join(child_path, dir_image))
                print(child_path)
                img_path = os.path.join(child_path, dir_image)
                img = nib.load(img_path)    #读取nii(91,109,91)
                y_temp =(0 if (child_path.split("\\")[-1]=="HC") else 1)
                img_fdata = img.get_fdata()

                for i in range(img_fdata.shape[2]):
                    #for i in range(40):
                    #if(startSlice+5<img_fdata.shape[2]):
                    temp_img = img_fdata[:,:,i]#(91,109)
#                    temp_img = temp_img/255.0
#                   temp_img = np.transpose(temp_img);
                    temp_img = cv2.resize(temp_img, (256, 192), interpolation=cv2.INTER_CUBIC)
                    temp_img = np.reshape(temp_img,(1,192,256))#以数组形式改变大小，加入到list#channel_first(th)
                    x_list.append(temp_img)
                    y_list.append(y_temp)
            if(n>=num):
                break
#将得到的图像（91*91）转换为
    return x_list,y_list

# path='D:/work/ADNI_AV45_TI_HC_MCI_AD/test/t1'
# x_train, _ = read_img(path)
# x_train =np.array(x_train)
# maxindex = np.unravel_index(np.argmax(x_train),x_train.shape)
# x_train/=x_train[maxindex]
# #plt.imshow(x_train[0,0,:,:])