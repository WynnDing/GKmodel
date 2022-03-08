"""
调用椎体检测模型
"""
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../数据处理'))
	print(os.getcwd())
except:
	pass

import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv.utils import download, viz
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

classes = ['bone']
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
net.load_parameters('ssd_test1.params')
save_path = '/media/baiqian/g/骨科治疗/输出图片/训练数据整理/骨节识别结果'
for file in path_list:
    #Dir = '/'.join(file.split('/')[:-1])
    Dir = os.path.join(save_path, file.split('/')[-2])
    if os.path.isdir(Dir):
        None
    else:
        os.mkdir(Dir)
    x, image = gcv.data.transforms.presets.ssd.load_test(file, 512)
    cid, score, bbox = net(x)


    #plt.imshow(image)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(1, 1, 1)
    ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes, ax=ax)
    plt.savefig(os.path.join(Dir, file.split('/')[-1]))
