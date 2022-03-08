"""
训练分类模型
"""
import mxnet as mx
import numpy as np
import os, time, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model

classes = 2

epochs = 100
lr = 0.00001
per_device_batch_size = 1
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
lr_steps = [10, 20, 30, np.inf]

num_gpus = 1
num_workers = 2
ctx = mx.gpu(0)

batch_size = 64
#ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
#batch_size = per_device_batch_size * max(num_gpus, 1)

jitter_param = 0.4
lighting_param = 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageRecordDataset('class.rec').transform_first(transform_train),
    batch_size=batch_size, shuffle=True)

for i in train_data:
    print(i[0].shape, i[1].shape)
    break

model_name = 'ResNet50_v2'
finetune_net = get_model(model_name, pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
                        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
metric = mx.metric.Accuracy()
#L = gluon.loss.SoftmaxCrossEntropyLoss()
L = gcv.loss.FocalLoss(num_class=2)

lr_counter = 0
num_batch = len(train_data)

for epoch in range(epochs):
    if epoch == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate*lr_factor)
        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, batch in enumerate(train_data):
        #data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        #label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        data, label = batch[0].as_in_context(ctx), batch[1].as_in_context(ctx)
        with ag.record():
            #outputs = [finetune_net(X) for X in data]
            #loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            
            outputs = finetune_net(data)
            loss = L(outputs, label)

            loss.backward()

        trainer.step(batch_size)
        train_loss += sum([loss.mean().asscalar()]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch

    #_, val_acc = test(finetune_net, val_data, ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.5f time: %.1f' %
             (epoch, train_acc, train_loss, time.time() - tic))

#_, test_acc = test(finetune_net, test_data, ctx)
#print('[Finished] Test-acc: %.3f' % (test_acc))

finetune_net.save_parameters('c1.params')