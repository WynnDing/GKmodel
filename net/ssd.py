"""
基于gluoncv的椎体识别模型
"""
import gluoncv

def model(model_path):
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
    net.load_parameters('ssd_test1.params')