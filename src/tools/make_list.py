#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 21:01:37 2019

@author: baiqian
解析标注数据将数据转换为，gluoncv固定读取格式.
"""
import os
import pandas as pd
import cv2

path = '/media/baiqian/g/骨科治疗/标注一类'

i = 0
path_list = []
files_list = []
for a, b, files in os.walk(path):
    if i != 0:
        for aa, bb, ff in os.walk(a):
            for f in ff:
                path_list.append(os.path.join(a, f))
                files_list.append(os.path.join(a.split('/')[-1], f))
    i+=1

df_path = pd.DataFrame({'label':path_list, 'text':files_list})
df_path = df_path[~df_path['label'].str.contains('classes')]
df_path = df_path[df_path['label'].str.contains('.txt')]
df_path['img'] = df_path['label'].str.replace('.txt', '.jpg')
df_path['text'] = df_path['text'].str.replace('.txt', '.jpg')


def write_line(img_path, im_shape, boxes, idx):
    h, w, c = im_shape
    A = 4
    B = 5
    C = w
    D = h
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = []
    for i in range(len(boxes)):
        labels = list(boxes.iloc[i])
        #labels[0] = int(labels[0])
        #print(labels)
        str_labels = str_labels + [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line

with open('train.lst', 'w') as fw:
    for i in range(len(df_path)):
        img = df_path['img'].iloc[i]
        img_path = df_path['text'].iloc[i]
        label_path = df_path['label'].iloc[i]
        df = pd.read_csv(label_path , sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
        #df = df[['label', 'x2', 'y2', 'x1', 'y1']]
        df['label'] = df['label'].apply(lambda x:int(x))
        df = df[df['label']==0]
        df['label']=1
        df['x1'] = df['x1']-df['x2']/2
        df['y1'] = df['y1']-df['y2']/2
        df['x2'] = df['x1']+df['x2']
        df['y2'] = df['y1']+df['y2']
        if len(df)>0:
            img = cv2.imread(img)
            im_shape = img.shape
            
            line = write_line(img_path, im_shape, df, i)
            fw.write(line)
        else:
            print('文件未标注')
        
