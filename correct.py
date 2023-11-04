import cv2
import numpy as np
import os

cv2.namedWindow('left')
cv2.namedWindow('right')

camera = cv2.VideoCapture(0)  # 打开摄像头，摄像头的ID不同设备上可能不同

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置双目的宽度(整个双目相机的图像宽度)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
while camera.isOpened():
    ret, frame = camera.read()
    left_image = frame[0:360, 0:640]  # 裁剪坐标为[y0:y1，x0：x1]
    right_image = frame[0:360, 640:1280]
    if not ret:
        break
    cv2.imshow('left', left_image)
    cv2.imshow('right', right_image)

    def cat2images(limg, rimg):
        HEIGHT = limg.shape[0]
        WIDTH = limg.shape[1]
        imgcat = np.zeros((HEIGHT, WIDTH*2+20, 3))
        imgcat[:, :WIDTH, :] = limg
        imgcat[:, -WIDTH:, :] = rimg
        for i in range(int(HEIGHT / 32)):
            imgcat[i*32, :, :] = 255
        return imgcat

    # left_image = cv2.imread("left_frame/22.jpg")
    # right_image = cv2.imread("right_frame/22.jpg")

    imgcat_source = cat2images(left_image, right_image)
    HEIGHT = left_image.shape[0]
    WIDTH = left_image.shape[1]
    # cv2.imwrite('imgcat_source.jpg', imgcat_source)

    k1 = np.array([4.8610516899917451e+02, 0., 2.4760875167424015e+02, 0.,
                   4.8630304775187938e+02, 2.1447914817490980e+02, 0., 0., 1.]
                  ) .reshape((3, 3))

    d1 = np.array([1.0811781281020096e-01, -1.4022465641707479e-01,
                   3.4437006353133827e-04, -8.5413258056917397e-04, 0.])

    R_l = np.array([9.9998468490930703e-01, 5.3341752685822155e-03,
                    1.4753037105431320e-03, -5.3337358316285811e-03,
                    9.9998573003700164e-01, -3.0163627647443870e-04,
                    -1.4768916387798495e-03, 2.9376277662404853e-04,
                    9.9999886624661660e-01]
                   ) .reshape((3, 3))
    P_l = np.array([4.8530335785822689e+02, 0., 2.5411099052429199e+02, 0., 0.,
                    4.8530335785822689e+02, 2.1369447898864746e+02, 0., 0., 0., 1.,
                    0.]
                   ) .reshape((3, 4))
    k2 = np.array([4.8445735881613331e+02, 0., 2.7204639716931285e+02, 0.,
                   4.8430366796457440e+02, 2.1233808715566130e+02, 0., 0., 1.]
                  ) .reshape((3, 3))

    d2 = np.array([9.4232478484657947e-02, -9.1328837797491735e-02,
                   -2.6901601339183686e-04, -9.2688284893052328e-04, 0.])

    R_r = np.array([9.9993847558626592e-01, 5.0683174716266502e-03,
                    9.8669752316342171e-03, -5.0715457488652678e-03,
                    9.9998709397028118e-01, 3.0218634846161733e-04,
                    -9.8653163118090498e-03, -3.5220857291384431e-04,
                    9.9995127455451016e-01]
                   ) .reshape((3, 3))
    P_r = np.array([4.8530335785822689e+02, 0., 2.5411099052429199e+02,
                    -2.9284159129569871e+04, 0., 4.8530335785822689e+02,
                    2.1369447898864746e+02, 0., 0., 0., 1., 0.]
                   ) .reshape((3, 4))
    Q = np.array([1., 0., 0., -2.5411099052429199e+02, 0., 1., 0.,
                  -2.1369447898864746e+02, 0., 0., 0., 4.8530335785822689e+02, 0.,
                  0., 1.6572214203281959e-02, 0.]
                 ) .reshape((4, 4))

    (map1, map2) = \
        cv2.initUndistortRectifyMap(k1, d1, R_l, P_l, np.array([
                                    WIDTH, HEIGHT]), cv2.CV_32FC1)  # 计算校正查找映射表
    left_imageG = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    rect_left_imageG = cv2.remap(
        left_imageG, map1, map2, cv2.INTER_CUBIC)  # 重映射

    # 左右图需要分别计算校正查找映射表以及重映射
    (map1, map2) = \
        cv2.initUndistortRectifyMap(
            k2, d2, R_r, P_r, np.array([WIDTH, HEIGHT]), cv2.CV_32FC1)
    right_imageG = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    rect_right_imageG = cv2.remap(right_imageG, map1, map2, cv2.INTER_CUBIC)
    # 转换为opencv的BGR格式
    rect_left_image = cv2.cvtColor(rect_left_imageG, cv2.COLOR_GRAY2BGR)
    rect_right_image = cv2.cvtColor(rect_right_imageG, cv2.COLOR_GRAY2BGR)
    imgcat_out = cat2images(rect_left_image, rect_right_image)
    # cv2.imwrite('imgcat_out.jpg', imgcat_out)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------

    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(rect_left_image, rect_right_image)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0,
                              beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16

    cv2.imshow('dis_color', dis_color)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    # cv2.imwrite('depth.jpg', dis_color)
    # cv2.imshow("depth", dis_color)
    # cv2.imshow("disparity", disp)  # 显示深度图的双目画面
    # while True:
    #     if cv2.waitKey(1) == ord('q'):
    #         break
