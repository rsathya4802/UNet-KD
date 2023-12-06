import torch
import cv2
import numpy as np
from PIL import Image

def split_squares(img, pos):
    h = img.shape[1]
    if(pos == 0):
        return img[:, :, :h]
    else:
        return img[:, :, -h:]

def normalize(img):
    return img/255

def hwc_to_chw(img):
    return np.transpose(img, (2, 0, 1))

def reduce_channel(img):
    if(img[:, :, 0] == img[:, :, 1] and img[:, :, 1] == img[:, :, 2]):
        return img[:, :, 0]

def load_data(img_path):
    if img_path.find("train") != -1:
        gt_path = img_path.replace("train", "train_mask")
    elif img_path.find("val") != -1:
        gt_path = img_path.replace("val", "val_mask")
    else:
        gt_path = img_path.replace("test", "test_mask")

    img = Image.open(img_path)#.resize((640, 959))
    gt = Image.open(gt_path)#.resize((640, 959))

    # cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # resized_image1 = cv2.resize(cv2_image, (384, 512), interpolation=cv2.INTER_NEAREST)[240:400]
    # resized_image2 = cv2.resize(resized_image1, (256, 256), interpolation=cv2.INTER_NEAREST)
    # img = Image.fromarray(cv2.cvtColor(resized_image2, cv2.COLOR_BGR2RGB))
    # # print(gt.size)
    # cv2_gt_image = cv2.cvtColor(np.array(gt), cv2.COLOR_RGB2BGR)
    # # print(cv2_gt_image.shape)
    # resized_mask1 = cv2.resize(cv2_gt_image, (384, 512), interpolation=cv2.INTER_NEAREST)[240:400]
    # # print(resized_mask1.shape)
    # resized_mask2 = cv2.resize(resized_mask1, (256, 256), interpolation=cv2.INTER_NEAREST)
    # # print(resized_mask2.shape)
    # gt = Image.fromarray(cv2.cvtColor(resized_mask2, cv2.COLOR_BGR2RGB))
    # # print(gt.size)

    return img, gt
    #add data aug functions
    #return img
        
