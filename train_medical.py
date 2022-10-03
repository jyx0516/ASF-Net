import time
import keras
import numpy as np
from nets.unet import Unet
from nets.unet_training_medical import Generator, dice_loss_with_CE, CE
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.metrics import categorical_accuracy
from keras import backend as K
from PIL import Image
from utils.metrics import Iou_score, f_score
import os


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    inputs_size = [512,512,3]
    log_dir = "logs/"
    #--------------------------------------------------------------------#
    #   简单的医药分割只分背景和边缘
    #--------------------------------------------------------------------#
    num_classes = 2
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = True

    # 获取model
    model = Unet(inputs_size,num_classes)
    # par_model = keras.utils.multi_gpu_model(model,gpus = 2)
    # model.summary()

    # model_path = "./model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model_path = "model_data/unet_medical.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open(r"./Medical_Datasets/ImageSets/Segmentation/train.txt","r") as f:
        train_lines = f.readlines()
        
    # 保存的方式，3世代保存一次
    checkpoint_period = ParallelModelCheckpoint(model,log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5',
                                    monitor='loss', save_weights_only=True, save_best_only=True, period=1)
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1)
    # tensorboard
    tensorboard = TensorBoard(log_dir=log_dir)

    if True:
        lr = 1e-4
        Freeze_Epoch = 0
        Unfreeze_Epoch = 100
        Batch_size = 2
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, with batch size {}.'.format(len(train_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes).generate()
        # 开始训练
        model.fit_generator(gen,
                steps_per_epoch=max(1, len(train_lines)//Batch_size),
                epochs=1000,
                initial_epoch=0,
                callbacks=[checkpoint_period, reduce_lr, tensorboard])
