import cv2
import time
import sys
from model import YOLOv5_plate

plate_det = YOLOv5_plate()

fcap = cv2.VideoCapture('video/face.mp4')

def add_mosaic(img, points, step):
    for point in points:
        x1 = point['x1']
        x2 = point['x2']
        y1 = point['y1']
        y2 = point['y2']

        for i in range(0, y2 - y1 - step, step):
            for j in range(0, x2 - x1 - step, step):
                color = img[i + y1][j + x1].tolist()
                cv2.rectangle(img, (x1 + j, y1 + i), (x1 + j + step - 1, y1 + i + step - 1), color, -1)

    return img

# 获取视频帧的宽
w = fcap.get(cv2.CAP_PROP_FRAME_WIDTH)
# 获取视频帧的高
h = fcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# 获取视频帧的帧率
fps = fcap.get(cv2.CAP_PROP_FPS)

# 获取VideoWriter类实例
writer = cv2.VideoWriter('video/output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (int(w), int(h)))

# 判断是否正确获取VideoCapture类实例
while fcap.isOpened():

    # 获取帧画面
    success, frame = fcap.read()

    cnt = 0
    start = time.time()
    while success:
        success, frame = fcap.read()

        _, points_plate = plate_det.infer(frame)
        frame = add_mosaic(frame, points_plate, 6)

        cv2.imwrite('video/pic/result' + str(cnt) + '.jpg', frame)
        # 保存帧数据
        writer.write(frame)

        cnt += 1
        if (cnt % 1000 == 0):
            print(cnt, "done")
            end = time.time()
            print('time cost', end - start,'s')
    
# 释放VideoCapture资源
fcap.release()
# 释放VideoWriter资源
writer.release()