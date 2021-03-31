"""
https://blog.csdn.net/sinat_29047129/article/details/103642140
https://www.cnblogs.com/Trevo/p/11795503.html
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵n*n，初始值全0

    # 像素准确率PA，预测正确的像素/总像素
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        PA = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return PA

    # 类别像素准确率CPA，返回n*1的值，代表每一类，包括背景
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # Pr = (TP) / TP + FP
        Pr = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        PR = self.confusionMatrix[0][0] / (self.confusionMatrix[0][0] + self.confusionMatrix[0][1])
        # print(self.confusionMatrix)
        # print(self.confusionMatrix[0][0])
        # print(self.confusionMatrix[0][1])
        # print(self.confusionMatrix[1][0])
        # print(self.confusionMatrix[1][1])
        #
        # print(np.diag(self.confusionMatrix))
        #
        # print(self.confusionMatrix.sum(axis=1))
        # print(self.confusionMatrix.sum())
        # print(Pr)
        # print(PR)
        return PR

    def Recall(self):
        # Re = (TP) / TP + FN
        Re = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        RE = self.confusionMatrix[0][0] / (self.confusionMatrix[0][0] + self.confusionMatrix[1][0])

        # return Re
        return RE

    def F1(self):
        # f1 = (2 * Pr * Re) / Pr + Re
        Pr = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        Pr = np.mean(Pr)
        Re = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        Re = np.mean(Re)
        RE = self.confusionMatrix[0][0] / (self.confusionMatrix[0][0] + self.confusionMatrix[0][1])
        # PR = self.confusionMatrix[0][0] / (self.confusionMatrix[0][0] + self.confusionMatrix[0][1])
        F1 = 2 * Pr * RE / (Pr + RE)
        return F1

    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    # 根据标签和预测图片返回其混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)

        #######
        confusionMatrix = np.flip(confusionMatrix, axis=0)
        confusionMatrix = np.flip(confusionMatrix, axis=1)
        #######
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU



    # 更新混淆矩阵
    def addBatch(self, imgPredict, imgLabel):

        assert imgPredict.shape == imgLabel.shape  # 确认标签和预测值图片大小相等

        # ################################
        imgPredict = imgPredict >= 20
        imgPredict = np.array(imgPredict).astype(int)
        imgLabel = np.array(imgLabel).astype(int)
        ################################

        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    # 清空混淆矩阵
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def old():
    imgPredict = np.array([0, 0, 0, 1, 2, 2])
    imgLabel = np.array([0, 0, 1, 1, 2, 2])
    metric = SegmentationMetric(2)
    metric.addBatch(imgPredict, imgLabel)
    PA = metric.pixelAccuracy()
    Pr = metric.Precision()
    Re = metric.Recall()
    F1 = metric.F1()
    mIoU = metric.meanIntersectionOverUnion()
    print(PA, Pr, Re, F1, mIoU)


def evaluate1(pre_path, label_path):
    PA_list = []
    Pr_list = []
    Re_list = []
    F1_list = []
    mIoU_list = []
    fwIoU_list = []

    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    for i, p in enumerate(pre_imgs):
        imgPredict = Image.open(pre_path + p)
        imgPredict = np.array(imgPredict)

        # imgPredict = imgPredict[:,:,0]

        # imgLabel = plt.imread(label_path + lab_imgs[i])
        imgLabel = Image.open(label_path + lab_imgs[i])
        imgLabel = np.array(imgLabel)

        # imgLabel = imgLabel[:,:,0]

        metric = SegmentationMetric(2)  # 表示分类个数，包括背景
        metric.addBatch(imgPredict, imgLabel)
        PA = metric.pixelAccuracy()
        Pr = metric.Precision()
        Re = metric.Recall()
        F1 = metric.F1()
        mIoU = metric.meanIntersectionOverUnion()
        fwIoU = metric.Frequency_Weighted_Intersection_over_Union()

        PA_list.append(PA)
        Pr_list.append(Pr)
        Re_list.append(Re)
        F1_list.append(F1)
        mIoU_list.append(mIoU)
        fwIoU_list.append(fwIoU)

        # print('{}: PA={}, Pr={}, mIoU={}, fwIoU={}'.format(p, acc, macc, mIoU, fwIoU))

    return PA_list, Pr_list, Re_list, F1_list, mIoU_list, fwIoU_list


def evaluate2(pre_path, label_path):
    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)
    metric = SegmentationMetric(2)  # 表示分类个数，包括背景
    for i, p in enumerate(pre_imgs):
        imgPredict = Image.open(pre_path + p)
        imgPredict = np.array(imgPredict)
        imgLabel = Image.open(label_path + lab_imgs[i])
        imgLabel = np.array(imgLabel)

        metric.addBatch(imgPredict, imgLabel)

    return metric


if __name__ == '__main__':
    pre_path = 'E:\XKF\dataset\pred_ECAnet/'
    label_path = 'E:\XKF\dataset\CFDtestdata\masks/'
    # 计算测试集每张图片的各种评价指标，最后求平均
    PA_list, Pr_list, Re_list, F1_list, mIoU_list, fwIoU_list = evaluate1(pre_path, label_path)
    print('final1: PA={:.2f}%, Pr={:.2f}%, Re={:.2f}%, F1={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%'
          .format(np.mean(PA_list) * 100, np.mean(Pr_list) * 100,
                  np.mean(Re_list) * 100, np.mean(F1_list) * 100,
                  np.mean(mIoU_list) * 100, np.mean(fwIoU_list) * 100))

    # # 加总测试机每张图片的混淆矩阵，对最终形成的这一个矩阵计算各种评价指标
    # metric = evaluate2(pre_path, label_path)
    # PA = metric.pixelAccuracy()
    # Pr = metric.Precision()
    # Re = metric.Recall()
    # F1 = metric.F1()
    # mIoU = metric.meanIntersectionOverUnion()
    # fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
    # print('final2: PA={:.2f}%, Pr={:.2f}%, Re={:.2f}%, F1={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%'
    #       .format(PA*100, Pr*100, Re*100, F1*100, mIoU*100, fwIoU*100))