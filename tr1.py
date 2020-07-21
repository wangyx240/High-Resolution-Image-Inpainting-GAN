import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torchsummary import summary
from torch.nn import functional as F

from network_module import *
import random

###########################################################################################################
#This script is used to verify the attention part.
###########################################################################################################
def generate_stroke_mask(im_size, parts=10, maxVertex=25, maxLength=60, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)

    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def cal_patch(patch_num, mask, raw_size):
    pool = nn.MaxPool2d(raw_size // patch_num)  # patch_num=32
    patch_fb = pool(mask)  # out: [B, 1, 32, 32]
    return patch_fb


def compute_attention(feature, patch_fb):  # in: [B, C:32, 64, 64]
    b = feature.shape[0]
    feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear')  # in: [B, C:32, 32, 32]
    p_fb = torch.reshape(patch_fb, [b, 32 * 32, 1])
    p_matrix = torch.bmm(p_fb, (1 - p_fb).permute([0, 2, 1]))
    f = feature.permute([0, 2, 3, 1]).reshape([b, 32 * 32, 32])
    c = cosine_Matrix(f, f) * p_matrix
    s = F.softmax(c, dim=2) * p_matrix
    return s


def attention_transfer( feature, attention):  # feature: [B, C, H, W]
    b_num, c, h, w = feature.shape
    f = extract_image_patches(feature, 32)
    f = torch.reshape(f, [b_num, f.shape[1] * f.shape[2], -1])
    f = torch.bmm(attention, f)
    f = torch.reshape(f, [b_num, 32, 32, h // 32, w // 32, c])
    x = f.permute([0, 5, 3, 1, 4, 2])
    x = torch.reshape(x, [b_num, c, h, w])
    y = f.permute([0, 5, 1, 3, 2, 4])
    y = torch.reshape(y, [b_num, c, h, w])
    res = x - y
    return y


def extract_image_patches( img, patch_num):
    b, c, h, w = img.shape
    img = torch.reshape(img, [b, c, patch_num, h//patch_num, patch_num, w//patch_num])
    img = img.permute([0, 2, 4, 3, 5, 1])
    return img


def cosine_Matrix( _matrixA, _matrixB):
    _matrixA_matrixB = torch.bmm(_matrixA, _matrixB.permute([0, 2, 1]))
    _matrixA_norm = torch.sqrt((_matrixA * _matrixA).sum(axis=2)).unsqueeze(dim=2)
    _matrixB_norm = torch.sqrt((_matrixB * _matrixB).sum(axis=2)).unsqueeze(dim=2)
    return _matrixA_matrixB / torch.bmm(_matrixA_norm, _matrixB_norm.permute([0, 2, 1]))

L1Loss = nn.L1Loss()
input = torch.rand(10,32,64,64)
# mask = np.array([generate_stroke_mask([256,256]) for i in range(10)])
# mask = torch.from_numpy(mask.astype(np.float32)).permute(0, 3, 1, 2).contiguous()
# patch_fb=cal_patch(32,mask,256)
# att = compute_attention(input,patch_fb)
x = torch.eye(1024) # 创建对角矩阵n*n
att = x.expand((10, 1024, 1024)) # 扩展维度到b维
out = attention_transfer(input, att)
res = input-out
print(L1Loss(input,out))

# su = np.array([
#     [2, 8, 7, 1, 6, 5],
#     [9, 5, 4, 7, 3, 2],
#     [6, 1, 3, 8, 4, 9],
#     [8, 7, 9, 6, 5, 1],
#     [4, 2, 1, 3, 9, 8],
#     [3, 6, 5, 4, 2, 7]
# ])
# aa = torch.from_numpy(su.astype(np.float32))
# fff = torch.reshape(aa, [3,2,3,2])
# zzz = torch.reshape(aa, [2,3,2,3])
# ff=fff.permute(0,2,1,3)
# print(zzz)
