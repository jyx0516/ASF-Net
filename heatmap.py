
# from keras.applications.vgg16 import VGG16
# # 特别注意，在之前的实验中，我们都把顶层的分类器丢弃掉了，include_top = False
# model = VGG16()
# print("模型调取成功")
#
from keras import backend as K
from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input, decode_predictions
# import numpy as np


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x
import os

import matplotlib.pyplot as plt
import numpy as np
from nets.unet import Unet as unet
input_shape     = [224, 224]
backbone = "resnet50"
num_classes = 2
alpha = 0.25
model_path = "logs/ep161-loss0.102.h5"
# ---------------------------------------------------#
#   载入模型与权值
# ---------------------------------------------------#

model = unet((512, 512, 3), num_classes)

model.load_weights(model_path)



# The local path to our target image
img_path = 'test/10.jpg'


# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(512, 512))

# 一转，`x` is a float32 Numpy array of shape (224, 224, 3)
x0 = image.img_to_array(img)

# 二扩，We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x1 = np.expand_dims(x0, axis=0)

# 三标，Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x1)
preds = model.predict(x)
# print('Predicted:', decode_predictions(preds, top=3)[0])
num = np.argmax(preds)#求最大的类别的索引
num=1
african_elephant_output = model.output[:, num]#获取索引为num的类的预测输出  shape: (batch_size,)
# last_conv_layer = model.get_layer('res5c_branch2c')#获取最后一个卷积层激活输出 shape (batch_size, 14, 14, 512)
last_conv_layer = model.get_layer('concatenate_2')#获取最后一个卷积层激活输出 shape (batch_size, 14, 14, 512)
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]#求模型输出针对最后一个卷积层激活输出的梯度 shape(batch_size,14,14,512)

#梯度均值化，即求各通道平均值，平均数,即对每一层 14 x 14的矩阵求均值, (batch_size,14,14, 512) ----> (512,)
pooled_grads = K.mean(grads, axis=(0, 1, 2))
print('pooled_grads:',pooled_grads.shape)
#建立模型输出、最后一个卷积层激活输出、梯度均值三者之间的函数关系
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# 以真实的数据作为输入，得到结果
pooled_grads_value, conv_layer_output_value = iterate([x])
print(pooled_grads_value.shape,conv_layer_output_value.shape)#(512,) (14, 14, 512)
##乘梯度
# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
#表征出最后卷积层激活输出各点对决策模型分类的重要程度。
for i in range(len(pooled_grads_value)):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1) # #shape:14*14
#Relu函数
heatmap = np.maximum(heatmap, 0)
#归一化处理
heatmap /= np.max(heatmap)  #shape:14*14
import matplotlib.pyplot as plt
plt.matshow(heatmap)
plt.show()

#读取原始图像
import cv2
test = cv2.imread(img_path)
#heatmap为[0,1]之间的浮点数，特别注意：cv2.resize(img, (x轴向长度，y轴向长度))
#调整热图尺寸，与原图保持一致，resize()
heatmap_test = cv2.resize(heatmap, (test.shape[1], test.shape[0]))
#可视化热力图
plt.matshow(heatmap_test)
plt.show()

#将heatmap数组转换为（0,255）之间的无符号的unit8数值
heatmap_test = np.uint8(255 * heatmap_test)
#将热力图转换为喷射效果
heatmap_test = cv2.applyColorMap(heatmap_test, cv2.COLORMAP_JET)
#将热力图与原始图像叠加， 0.5表示渲染强度, 有超出（0,255）范围的，如果需要可视化，则需要clip裁剪
superimposed_img_test = heatmap_test * 0.5 + test
superimposed_img_test=np.clip(superimposed_img_test,0,255)
print(np.max(superimposed_img_test),superimposed_img_test.shape)
superimposed_img_test=superimposed_img_test.astype(np.uint8) ##必须做，要不然会白屏
#用OpenCV中imread输入照片后是一个数组对象，在进行一系列的对数组操作后数组已经变成了float类型，之后再对数组进行imshow时即出现上面的第二种情况。倘若图像矩阵（double型）的矩阵元素不在0-1之间，那么imshow会把超过1的元素都显示为白色，即255。其实也好理解，因为double的矩阵并不是归一化后的矩阵并不能保证元素范围一定就在0-1之间，所以就会出错。
cv2.imshow('1',superimposed_img_test)
cv2.waitKey(0)
cv2.imwrite('a.jpg',superimposed_img_test)#写