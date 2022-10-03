#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from unet import Unet
from PIL import Image
import cv2

import os




os.environ["CUDA_VISIBLE_DEVICES"] = "0"
unet = Unet()

def get_file_list(path):
    fileslist = os.listdir(path)
    # 先定义一个排序的空列表
    sort_num_list = []
    for file in fileslist:
        sort_num_list.append(int(file.split('.jpg')[0]))  # 去掉前面的字符串和下划线以及后缀，只留下数字并转换为整数方便后面排序
        sort_num_list.sort()  # 然后再重新排序

    # print(sort_num_list)
    # 接着再重新排序读取文件
    sorted_file = []
    for sort_num in sort_num_list:
        for file in fileslist:
            if str(sort_num) == file.split('.jpg')[0]:
                sorted_file.append(file)
    return sorted_file
    # # print(sorted_file)
    # for i in sorted_file:
    #     file = path + '/' + i
    #     print(file)

list = get_file_list("test")

index = 0
for name in list:
    img_name = "test\\"+name
    print(img_name)
    image = Image.open(img_name)
    index += 1
    r_image = unet.detect_image(image, index)
    r_image.save("test_mask_result\\"+str(index)+".jpg")
# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = unet.detect_image(image)
#         # r_image.save("new.jpg")
#         r_image.show()
