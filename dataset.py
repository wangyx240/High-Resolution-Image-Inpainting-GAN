import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import utils

ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']

class InpaintDataset(Dataset):
    def __init__(self, opt):
        assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)
        #self.imglist=self.imglist[0:300]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        img = cv2.imread(self.imglist[index])
        if self.imglist[index]:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                print(self.imglist[index])
            img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize))
        else:
            print(self.imglist[index],11)
        # mask
        if self.opt.mask_type == 'single_bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        if self.opt.mask_type == 'bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        if self.opt.mask_type == 'free_form':
            #mask = self.random_ff_mask(shape = self.opt.imgsize, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)
            mask = self.generate_stroke_mask([self.opt.imgsize, self.opt.imgsize])
        
        # the outputs are entire image and mask, respectively
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1).contiguous()

        return img, mask

    def generate_stroke_mask(self, im_size, parts=7, maxVertex=25, maxLength=80, maxBrushWidth=80, maxAngle=360):
        mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
        for i in range(parts):
            mask = mask + self.np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        mask = np.minimum(mask, 1.0)

        return mask

    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
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

    # def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
    #     """Generate a random free form mask with configuration.
    #     Args:
    #         config: Config should have configuration including IMG_SHAPES,
    #             VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    #     Returns:
    #         tuple: (top, left, height, width)
    #     """
    #     height = shape
    #     width = shape
    #     mask = np.zeros((height, width), np.float32)
    #     times = np.random.randint(times)
    #     for i in range(times):
    #         start_x = np.random.randint(width)
    #         start_y = np.random.randint(height)
    #         for j in range(1 + np.random.randint(5)):
    #             angle = 0.01 + np.random.randint(max_angle)
    #             if i % 2 == 0:
    #                 angle = 2 * 3.1415926 - angle
    #             length = 10 + np.random.randint(max_len)
    #             brush_w = 5 + np.random.randint(max_width)
    #             end_x = (start_x + length * np.sin(angle)).astype(np.int32)
    #             end_y = (start_y + length * np.cos(angle)).astype(np.int32)
    #             cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
    #             start_x, start_y = end_x, end_y
    #     return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
        
class ValidationSet_with_Known_Mask(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.namelist = utils.get_names(opt.baseroot)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        # image
        imgname = self.namelist[index]
        imgpath = os.path.join(self.opt.baseroot, imgname)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize))
        # mask
        maskpath = os.path.join(self.opt.maskroot, imgname)
        img = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        # the outputs are entire image and mask, respectively
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return img, mask, imgname
