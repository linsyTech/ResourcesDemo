# 调用双目摄像头录像，按q退出录制

import cv2
import time
import os

cv2.namedWindow('left')
cv2.namedWindow('right')

camera = cv2.VideoCapture(0)  # 打开摄像头，摄像头的ID不同设备上可能不同

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置双目的宽度(整个双目相机的图像宽度)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # 设置双目的高度

fourcc = cv2.VideoWriter_fourcc(*'H264')       # fourcc编码为视频格式，avi对应编码为XVID
# (视频储存位置和名称，储存格式，帧率，视频大小)
# left = cv2.VideoWriter("zuo.mp4", fourcc, 20.0, (1280, 960))
# right = cv2.VideoWriter("you.mp4", fourcc, 20.0, (1280, 960))

while camera.isOpened():
    ret, frame = camera.read()
    left_frame = frame[0:360, 0:640]  # 裁剪坐标为[y0:y1，x0：x1]
    right_frame = frame[0:360, 640:1280]
    if not ret:
        break

    # left.write(left_frame)
    # right.write(right_frame)
    cv2.imshow('left', left_frame)
    cv2.imshow('right', right_frame)
    key = cv2.waitKey(1)
    if key == ord('c'):
        # 先判断文件夹里有没有文件，有的话就给文件名加上当前文件及内的文件数量
        if not os.path.exists('left_frame'):
            os.mkdir('left_frame')
        if not os.path.exists('right_frame'):
            os.mkdir('right_frame')
        count_left = len(os.listdir('left_frame'))
        count_right = len(os.listdir('right_frame'))
        cv2.imwrite('left_frame/' + str(count_left) + '.jpg', left_frame)
        cv2.imwrite('right_frame/' + str(count_right) + '.jpg', right_frame)
        print("已保存当前帧")

    elif key == ord('q'):
        break


camera.release()
# left.release()
# right.release()
cv2.destroyAllWindows()
