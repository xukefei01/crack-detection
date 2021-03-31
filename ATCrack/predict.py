from __future__ import division
from pathlib import Path
import os
import sys
import time
import datetime
import argparse
import cv2 as cv
from PIL import Image
import numpy as np
import torch
from unet.unet_transfer import input_size
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from os.path import join
from utils import load_unet_vgg16
from tqdm import tqdm
import shutil
from unet.unet_transfer import UNet16
import gc


def predict_init():
    model_path = 'weights/vgg16_66.pt'
    start = time.time()
    model = load_unet_vgg16(model_path)
    end = time.time()
    print('model load time:', end-start)

    return model


def predict_segment(model, img_dir='upload/images/'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir', type=str, default='upload/images/', help='input dataset directory')
    parser.add_argument('-model_path', type=str, default="weights/vgg16_66.pt", help='trained model path')
    parser.add_argument('-model_type', type=str, default='vgg16', choices=['vgg16'])
    parser.add_argument('-out_pred_dir', type=str, default='static/images/', required=False,  help='prediction output dir')
    parser.add_argument('-threshold', type=float, default=0.2, help='threshold to cut off crack response')
    args = parser.parse_args()
    print(args)

    start = time.time()
    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))

    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    else:
        print('undefind model name pattern')
        exit()

    end = time.time()
    print('model predict time:', end-start)
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
    paths = [path for path in Path(args.img_dir).glob('*.*')]
    for path in tqdm(paths):

        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)

        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:, :, :3]

        img_height, img_width, img_channels = img_0.shape
        input_width, input_height = input_size[0], input_size[1]
        img_1 = cv.resize(img_0, (input_width, input_height), cv.INTER_AREA)
        X = train_tfms(Image.fromarray(img_1))
        X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]
        mask = model(X)

        mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
        mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
        prob_map_full = mask

        if args.out_pred_dir != '':
            cv.imwrite(filename=join(args.out_pred_dir, f'predict.jpg'), img=(prob_map_full * 255).astype(np.uint8))

    path_data = r"E:\deeplearn-model\AttU_net_master\upload\images"
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)

        gc.collect()


if __name__ == '__main__':
    model = predict_init()
    predict_segment(model=model)

