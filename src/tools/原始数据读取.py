#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 15:16:47 2018
@author: dong
将原始数据转为jpg格式图片
"""
import os
import cv2
import numpy as np
import SimpleITK as sitk
import pylab as plt
from PIL import Image
from skimage import data, io, filters

#filename = "/home/dong/项目/骨科治疗/电子阅片-试验/曹连娣/10001.dcm"

if __name__ == "__main__":
    path = '/home/dong/项目/骨科治疗/原始数据'
    save_path = '/home/dong/项目/骨科治疗/标注图片'
    path_list = []
    files_list = []

    for a, b, files in os.walk(path):
        for aa, bb, ff in os.walk(a):
            for f in ff:
                path_list.append(aa)
                files_list.append(f)

    for i in range(len(files_list)):
        file = files_list[i]
        p = path_list[i]
        pp = p.split('/')[-1]
        save = os.path.join(save_path, pp, '%s.jpg' % file.split('.')[0])
        try:
            ds = sitk.ReadImage(os.path.join(p, file))
            img_array = sitk.GetArrayFromImage(ds)
            frame_num, width, height = img_array.shape
            img = img_array[0]
            img = np.array(img)
            if os.path.isdir(os.path.join(save_path, pp)):
                io.imsave(save, img/4000)
            else:
                os.mkdir(os.path.join(save_path, pp))
                io.imsave(save, img/4000)
        except:
            print(os.path.join(p, file))

    #ds = sitk.ReadImage('/home/dong/项目/骨科治疗/原始数据/张熠1/10012.dcm')
