# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device

def show_results(img, xywh, conf):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

class YOLOv5_plate(object):
    # 参数设置
    _defaults = {
        "weights": "runs/train/exp4/weights/best.pt",
        "imgsz": 640,
        "iou_thres":0.45,
        "conf_thres":0.24,
        "classes":1 
    }

    @classmethod
    def get_defaults(cls,n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化操作，加载模型
    def __init__(self,device='0',**kwargs):
        self.__dict__.update(self._defaults)
        self.device = select_device(device)
        self.half = self.device != "cpu" 

        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

    def infer(self,inImg):
        # 使用letterbox方法将图像大小调整为640大小
        img = letterbox(inImg, new_shape=self.imgsz)[0]

        # 归一化与张量转换
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        pred = self.model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        # print('img.shape: ', img.shape)
        # print('inImg.shape: ', inImg.shape)

        h, w, c = inImg.shape
        
        id = []
        category = []
        points = []

        # 解析检测结果
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(inImg.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain whwh
            gn_lks = torch.tensor(inImg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(self.device)  # normalization gain landmarks
            
            if det is not None and len(det):
                # 将检测框映射到原始图像大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], inImg.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                
                
                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy() # 检测框文字
                    inImg = show_results(inImg, xywh, conf)
                    
                    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
                    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
                    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
                    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
                    point = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

                    id.append(j)
                    category.append(1)
                    points.append(point)
            else:
                print("None")

        cv2.imwrite('result.jpg', inImg)

        return category, points