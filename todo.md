# 工作计划及实施记录
骨科诊疗模型开发记录，与技术路线摸索
____
## 0426
## 工作计划
* 数据处理部分，将T2数据单独整理出来(完成);
* 调用之前椎体识别模型对T2数据椎体部分进行提取，根据检测结果的最大边，截取正方形数据，并解决部分靠近边缘的椎体提取失败的问题(完成); 
* 统一调整为标准网络(未完成); 
* 借用ssp网络结构，每个batch采用不同的尺寸对模型进行训练，测试效果(未完成);
## 完成情况
## 遗留问题
* 部分dom文件未读取
* 部分t2数据表现不一致（张熠1）ct图像类别
* 发现同一ct明显区别与其他组织
* 与周边组织有关

------
## 0503
## 工作计划
* 完成分类模型数据准备

## 遗留问题
* 部分dom文件未读取
* 部分t2数据表现不一致（张熠1）ct图像类别
* 发现同一ct明显区别与其他组织
* 与周边组织有关
* 曹连媂的200开始图片灰度明显不同识别难度较大，所以从分类网络里面剔除