"""
椎体检测
"""
import os
# try:
# 	os.chdir(os.path.join(os.getcwd(), '../../数据处理'))
# 	print(os.getcwd())
# except:
# 	pass
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
import gluoncv as gcv

class Detect:
	"""椎体检测"""
	def __init__(self, model_path, data_path):
		"""
			
		"""
		self.model_path = model_path
		self.data_path = data_path
		#self.save_path = save_path

	def get_file(self):
		"""
			获取图片地址
		"""
		path = self.data_path
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
		return path_list

	def plot_bone(self, save_path):
		"""
			保存整张椎体识别结果图片
		"""
		path_list = self.get_file()
		classes = ['bone']
		net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
		net.load_parameters(self.model_path)

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

	def excert_bone(self, save_path):
		"""
			提取椎体区域
		"""
		path_list = self.get_file()
		classes = ['bone']
		net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
		net.load_parameters(self.model_path)
		#save_path = '/media/baiqian/g/骨科治疗/输出图片/训练数据整理/骨节区域提取'
		for file in path_list:
			x, image = gcv.data.transforms.presets.ssd.load_test(file, 512)
			cid, score, bbox = net(x)
			s = (score[0]>0.5).sum()
			s = s.asnumpy()[0]
			
			file_name = file.split('/')[-2]
			idx = file.split('/')[-1].split('.')[0]
			
			for i in range(int(s)):
				num = score[0][i].asnumpy()
				num = str(num[0])
				
				candidate = bbox[0][i].asnumpy()
				x_min, y_min, x_max, y_max =  candidate[0], candidate[1], candidate[2], candidate[3]
				center = ((x_min+x_max)/2, (y_min+y_max)/2)
				long_edge = max([x_max-x_min, y_max-y_min])
				# 图片切割为正方形				
				bone_edge = [center[0]-long_edge/2, center[1]-long_edge/2,
							center[0]+long_edge/2, center[1]+long_edge/2]
				#print(image.shape)
				if min(bone_edge) <= 0: # 判断方框是否超出边界
					# print(x_min, y_min, x_max, y_max) 将y轴整体平移
					img = image[int(bone_edge[1]-bone_edge[1]):int(bone_edge[3]-bone_edge[1]),
						 int(bone_edge[0]):int(bone_edge[2])]
				elif max(bone_edge) >= image.shape[1]:
					diff = bone_edge[3]-image.shape[1]
					img = image[int(bone_edge[1]-diff):int(bone_edge[3]-diff),
						 int(bone_edge[0]):int(bone_edge[2])]
				else:
					# [y_min: y_max, x_min:x_max]
					img = image[int(bone_edge[1]):int(bone_edge[3]),
						 int(bone_edge[0]):int(bone_edge[2])]
				if min(img.shape) <= 0:
					print(img.shape)
				cv2.imwrite(os.path.join(save_path, '%s_%s_%s.jpg' % (file_name, idx, num)), img)

if __name__ == "__main__":
	model_path = '/home/baiqian/project/GK/code/GKmodel/models/ssd_test1.params'
	data_path = '/home/baiqian/project/GK/data/T2'
	detect = Detect(model_path, data_path)
	
	save_path = '/home/baiqian/project/GK/data/椎体提取'
	try:
		os.mkdir(save_path)
	except:
		print('文件夹已经存在！')
	#detect.plot_bone(save_path)
	detect.excert_bone(save_path)