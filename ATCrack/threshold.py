"""
cv2.Canny(image,            # 输入原图（必须为单通道图）
          threshold1,
          threshold2,       # 较大的阈值2用于检测图像中明显的边缘
          [, edges[,
          apertureSize[,    # apertureSize：Sobel算子的大小
          L2gradient ]]])   # 参数(布尔值)：
                              true： 使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开放），
                              false：使用L1范数（直接将两个方向导数的绝对值相加）。
"""

import cv2
import numpy as np
from matplotlib import pyplot as plot
import os


root_path = 'E:\dataset\work_img'
# os.mkdir("E:/dataset/threshold_pre")            # 建立新的目录
out_path ="E:/dataset/work_img_threshold"
fileList = os.listdir(root_path)
for file in fileList:

    filepath = os.path.join(root_path, file)
    print('文件名路径:{}'.format(filepath))
    ori_img = cv2.imread(filepath, 0)
    # gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    # canny(): 边缘检测
    img = cv2.medianBlur(ori_img, 5)
    ret, img_pre=cv2.threshold(img,127,255,cv2.THRESH_BINARY) #阈值化
    img = [img_pre]
    # img = cv2.GaussianBlur(ori_img, (3, 3), 0)
    # img_pre = cv2.Canny(img, 50, 150)
    # img = [img_pre]
    for i in range(1):
        plot.subplot(1,1,i+1)
        plot.axis('off')
        plot.imshow(img[i],'gray')
        cv2.imwrite((os.path.join(out_path, file)), img[i])

    # save_path = os.path.join('E:/dataset/cannypre', file)
    # plot.savefig(save_path)
