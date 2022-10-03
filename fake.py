
TP = 53064421
FP = 8484144
TN = 179318014
FN = 9253421

print("TP: "+str(TP))
print("FP: "+str(FP))
print("TN: "+str(TN))
print("FN: "+str(FN))

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


# 总像素数：250000000
# TP+TN+FP+FN250000000
# 像素准确率（Pixel Accuracy，PA）: 0.941158948
# 类别像素准确率（Class Pixel Accuray，CPA）:正例0.882490795530847
# 类别像素准确率（Class Pixel Accuray，CPA）:反例0.9606040267923799
# 类别平均像素准确率（Mean Pixel Accuracy，MPA）:0.9215474111616134
# 交并比（Intersection over Union，IoU）:0.78873933579973
# 平均交并比（Mean Intersection over Union，MIoU）:0.8566663962417103
#
# Process finished with exit code 0
