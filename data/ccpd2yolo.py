import cv2
import os
from tqdm import tqdm
 
origin_CCPD20_path = "/data/dataset/CCPD2020/ccpd_green/"
origin_CCPD19_path = "/data/dataset/CCPD2019/"
save_path = "/data/dataset/CCPD/"
save_images_path = save_path + "images/"
save_labels_path = save_path + "labels/"

for sub_dir in ["ccpd_base/", "ccpd_fn/", "ccpd_db/", "ccpd_rotate/", "ccpd_tilt/", "ccpd_weather/", "ccpd_challenge/", "ccpd_blur/", "ccpd_np/"]:
    dir = os.listdir(origin_CCPD19_path + sub_dir)
    file_num = len(dir)
    cnt = 0
    for file_name in tqdm(dir):
        cnt += 1
        tag = cnt / file_num
        if tag <= 0.8: split = "train/"
        elif tag > 0.9: split = "val/"
        else: split = "test/"

        if sub_dir != "ccpd_np/":
            list1 = file_name.split("-", 3)  # 第一次分割，以减号'-'做分割
            subname = list1[2]
            lt, rb = subname.split("_", 1) #第二次分割，以下划线'_'做分割
            lx, ly = lt.split("&", 1)
            rx, ry = rb.split("&", 1)
            width = int(rx) - int(lx)
            height = int(ry) - int(ly)  # bounding box的宽和高
            cx = float(lx) + width/2
            cy = float(ly) + height/2 #bounding box中心点

            img = cv2.imread(origin_CCPD19_path + sub_dir + file_name)
            cv2.imwrite(save_images_path + split + file_name, img)
            width = width/img.shape[1]
            height = height/img.shape[0]
            cx = cx/img.shape[1]
            cy = cy/img.shape[0]

            txtname = file_name.split(".", 1)
            txtfile = save_labels_path + split + txtname[0] + ".txt"
            with open(txtfile, "w") as f:
                f.write(str(0)+" "+str(cx)+" "+str(cy)+" "+str(width)+" "+str(height))
        else:
            img = cv2.imread(origin_CCPD19_path + sub_dir + file_name)
            cv2.imwrite(save_images_path + split + file_name, img)

            txtname = file_name.split(".", 1)
            txtfile = save_labels_path + split + txtname[0] + ".txt"
            f = open(txtfile, "w")
            f.close()

    print(sub_dir, "done!")

for split in ["train/", "val/", "test/"]:
    for file_name in tqdm(os.listdir(origin_CCPD20_path + split)):
        list1 = file_name.split("-", 3)  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        lt, rb = subname.split("_", 1) #第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width/2
        cy = float(ly) + height/2 #bounding box中心点
 
        img = cv2.imread(origin_CCPD20_path + split + file_name)
        cv2.imwrite(save_images_path + split + file_name, img)
        width = width/img.shape[1]
        height = height/img.shape[0]
        cx = cx/img.shape[1]
        cy = cy/img.shape[0]
 
        txtname = file_name.split(".", 1)
        txtfile = save_labels_path + split + txtname[0] + ".txt"
        with open(txtfile, "w") as f:
            f.write(str(0)+" "+str(cx)+" "+str(cy)+" "+str(width)+" "+str(height))

    print(split, "done!")