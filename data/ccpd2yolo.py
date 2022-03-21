import cv2
import os
from tqdm import tqdm
 
images_path = "/data/dataset/CCPD2020/images/"
labels_path = "/data/dataset/CCPD2020/labels/"

for split in ["train/", "val/", "test/"]:
    for filename in tqdm(os.listdir(images_path + split)):
        list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        lt, rb = subname.split("_", 1) #第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width/2
        cy = float(ly) + height/2 #bounding box中心点
 
        img = cv2.imread(images_path + split + filename)
        width = width/img.shape[1]
        height = height/img.shape[0]
        cx = cx/img.shape[1]
        cy = cy/img.shape[0]
 
        txtname = filename.split(".", 1)
        txtfile = labels_path + split + txtname[0] + ".txt"
        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, "w") as f:
            f.write(str(0)+" "+str(cx)+" "+str(cy)+" "+str(width)+" "+str(height))

    print(split, "done!")