import cv2
import os
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

gt_folder = "gt_eval"
res_folder = "test_result_add"

gt_list = getFileList(gt_folder, [], 'jpg')
res_list = getFileList(res_folder, [], 'jpg')

TP = 0 # 真正例
FP = 0 # 假正例
TN = 0 # 真反例
FN = 0 # 假反例

for index in range(len(gt_list)):
    print(gt_list[index])
    print(res_list[index])
    test = cv2.imread(res_list[index], cv2.COLOR_BGR2GRAY)
    gt = cv2.imread(gt_list[index], cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    retval, test = cv2.threshold(test, 127, 255, cv2.THRESH_BINARY)
    for i in range(5000):
        for j in range(5000):
            # test为白色 gt为白色     真正例
            if test[i][j] == dst[i][j] and test[i][j] == 255:
                TP += 1
            # test为白色 gt为黑色     假正例
            if test[i][j] != dst[i][j] and test[i][j] == 255 and dst[i][j] == 0:
                FP += 1
            # test为黑色 gt为黑色     真反例
            if test[i][j] == dst[i][j] and test[i][j] == 0:
                TN += 1
            # test为黑色 gt为白色     假反例
            if test[i][j] != dst[i][j] and test[i][j] == 0 and dst[i][j] == 255:
                FN += 1

print("TP: "+str(TP))
print("FP: "+str(FP))
print("TN: "+str(TN))
print("FN: "+str(FN))
# print("####################################################")
#
# test = cv2.imread("test.png", cv2.COLOR_BGR2GRAY)
# gt = cv2.imread("gt.jpg", cv2.COLOR_BGR2GRAY)
# test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
# retval, dst = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
# print(dst.shape)
# print(test.shape)
#
# # 255 白 0 黑
#
# TP = 0 # 真正例
# FP = 0 # 假正例
# TN = 0 # 真反例
# FN = 0 # 假反例
# for i in range(512):
#     for j in range(512):
#         # test为白色 gt为白色     真正例
#         if test[i][j] == dst[i][j] and test[i][j] == 255:
#             TP += 1
#         # test为白色 gt为黑色     假正例
#         if test[i][j] != dst[i][j] and test[i][j] == 255 and dst[i][j] == 0:
#             FP += 1
#         # test为黑色 gt为黑色     真反例
#         if test[i][j] == dst[i][j] and test[i][j] == 0:
#             TN += 1
#         # test为黑色 gt为白色     假反例
#         if test[i][j] != dst[i][j] and test[i][j] == 0 and dst[i][j] == 255:
#             FN += 1
# print("TP: "+str(TP))
# print("FP: "+str(FP))
# print("TN: "+str(TN))
# print("FN: "+str(FN))
print("总像素数："+str(5000*5000*10))
print("TP+TN+FP+FN"+str(TP+TN+FP+FN))
mpa = (TP/(TP+FP))+(TN/(TN+FN))
mpa /= 2
print("像素准确率（Pixel Accuracy，PA）: "+str((TP + TN)/(TP + TN + FP + FN)))
print("类别像素准确率（Class Pixel Accuray，CPA）:正例"+str(TP / (TP + FP)))
print("类别像素准确率（Class Pixel Accuray，CPA）:反例"+str(TN / (TN + FN)))
print("类别平均像素准确率（Mean Pixel Accuracy，MPA）:"+str(mpa))
print("交并比（Intersection over Union，IoU）:"+str(TP / (TP + FP + FN)))
miou = TP / (TP + FP + FN) + TN / (TN + FN + FP)
miou /= 2
print("平均交并比（Mean Intersection over Union，MIoU）:"+str(miou))
# cv2.imshow("1", gt)
# cv2.imshow("2", dst)
# cv2.waitKey(0)
i = 0