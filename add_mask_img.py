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



input_folder = "test_mask_result"
out_folder = "test_mask_result_add"

# gt_list = getFileList(gt_folder, [], 'jpg')
gt_list = get_file_list(input_folder)
# for name in gt_list:
#     file_name = file = gt_folder + '\\' + name;
    # print(file_name)
boxes =[]
img = cv2.imread(r"gt_eval\1.jpg")
for i in range(len(gt_list)):
    file_name = file = input_folder + '\\' + gt_list[i];
    box = cv2.imread(file_name)
    boxes.append(box)
index = 0
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\1.jpg", img)


index = 25
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\2.jpg", img)

index = 50
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\3.jpg", img)

index = 75
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\4.jpg", img)

index = 100
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\5.jpg", img)

index = 125
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\6.jpg", img)


index = 150
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\7.jpg", img)
#

index = 175
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\8.jpg", img)
#


index = 200
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\9.jpg", img)
#


index = 225
for i in range(5):
    for j in range(5):
        img[i*1000:(i+1)*1000, j*1000:(j+1)*1000] = boxes[index]
        index += 1
# img = cv2.resize(img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("add", img)
cv2.imwrite(out_folder+"\\10.jpg", img)


# cropped = img[0:1000, 0:1000] # 裁剪坐标为[y0:y1, x0:x1]
# cropped2 = img[0:1000, 1000:2000] # 裁剪坐标为[y0:y1, x0:x1]
# cropped2 = img[0:1000, 2000:3000] # 裁剪坐标为[y0:y1, x0:x1]
# cropped2 = img[0:1000, 3000:4000] # 裁剪坐标为[y0:y1, x0:x1]
# cropped2 = img[0:1000, 4000:5000] # 裁剪坐标为[y0:y1, x0:x1]
# cropped2 = img[1000:2000, 0:2000] # 裁剪坐标为[y0:y1, x0:x1]
# cv2.imwrite("1.jpg", cropped)
# cv2.imwrite("2.jpg", cropped2)