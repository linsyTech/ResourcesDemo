import cv2
import numpy as np


def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH*2+20, 3))
    imgcat[:, :WIDTH, :] = limg
    imgcat[:, -WIDTH:, :] = rimg
    for i in range(int(HEIGHT / 32)):
        imgcat[i*32, :, :] = 255
    return imgcat


left_image = cv2.imread("left_frame/30.jpg")
right_image = cv2.imread("right_frame/30.jpg")

imgcat_source = cat2images(left_image, right_image)
HEIGHT = left_image.shape[0]
WIDTH = left_image.shape[1]
cv2.imwrite('imgcat_source.jpg', imgcat_source)

k1 = np.array([7.2988587825395541e+02, 0., 5.3768266667183195e+02, 0.,
               7.2924199083951612e+02, 3.1391400119509535e+02, 0., 0., 1.]
              ) .reshape((3, 3))

d1 = np.array([7.7968031794572068e-02, -9.4908512850966423e-02,
               1.2523461417107108e-03, -1.8815363908536423e-03, 0.])

R_l = np.array([9.9995126984656713e-01, -9.6493684437018445e-04,
                -9.8248068237634378e-03, 9.4171929431276235e-04,
                9.9999675403022725e-01, -2.3675079260652250e-03,
                9.8270594283646640e-03, 2.3581403468919018e-03,
                9.9994893273461527e-01]
               ) .reshape((3, 3))
P_l = np.array([7.2324862686443566e+02, 0., 5.6597463810443878e+02, 0., 0.,
                7.2324862686443566e+02, 3.7821684861183167e+02, 0., 0., 0., 1.,
                0.]
               ) .reshape((3, 4))
k2 = np.array([7.1763601164297631e+02, 0., 5.6960166595352507e+02, 0.,
               7.1725526288935532e+02, 3.2045944453541131e+02, 0., 0., 1.]
              ) .reshape((3, 3))

d2 = np.array([1.1524430041115404e-01, -1.4574783818062309e-01,
               7.6172754411533907e-04, -1.1155450427045254e-03, 0.])

R_r = np.array([9.9999861247877875e-01, -7.5642062156944202e-04,
                -1.4842063066575406e-03, 7.5993264593793948e-04,
                9.9999690959687926e-01, 2.3671288652327149e-03,
                1.4824111747741642e-03, -2.3682534776169164e-03,
                9.9999609690867042e-01]
               ) .reshape((3, 3))
P_r = np.array([7.2324862686443566e+02, 0., 5.6597463810443878e+02,
                -4.3600283373628372e+04, 0., 7.2324862686443566e+02,
                3.7821684861183167e+02, 0., 0., 0., 1., 0.]
               ) .reshape((3, 4))
Q = np.array([1., 0., 0., -5.6597463810443878e+02, 0., 1., 0.,
              -3.7821684861183167e+02, 0., 0., 0., 7.2324862686443566e+02, 0.,
              0., 1.6588163445329634e-02, 0.]
             ) .reshape((4, 4))
(map1, map2) = \
    cv2.initUndistortRectifyMap(k1, d1, R_l, P_l, np.array([
                                WIDTH, HEIGHT]), cv2.CV_32FC1)  # 计算校正查找映射表

rect_left_image = cv2.remap(left_image, map1, map2, cv2.INTER_CUBIC)  # 重映射

# 左右图需要分别计算校正查找映射表以及重映射
(map1, map2) = \
    cv2.initUndistortRectifyMap(
        k2, d2, R_r, P_r, np.array([WIDTH, HEIGHT]), cv2.CV_32FC1)

rect_right_image = cv2.remap(right_image, map1, map2, cv2.INTER_CUBIC)

imgcat_out = cat2images(rect_left_image, rect_right_image)
cv2.imwrite('imgcat_out.jpg', imgcat_out)


# ------------------------------------SGBM算法----------------------------------------------------------
#   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
#   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
#   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
#                               取16、32、48、64等
#   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
#                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
# ------------------------------------------------------------------------------------------------------


blockSize = 11
img_channels = 3
stereo = cv2.StereoSGBM_create(minDisparity=1,
                               numDisparities=64,
                               blockSize=blockSize,
                               P1=8 * img_channels * blockSize * blockSize,
                               P2=32 * img_channels * blockSize * blockSize,
                               disp12MaxDiff=1,
                               preFilterCap=1,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=2,
                               mode=cv2.STEREO_SGBM_MODE_HH)
# 计算视差
disparity = stereo.compute(rect_left_image, rect_right_image)

# 归一化函数算法，生成深度图（灰度图）
disp = cv2.normalize(disparity, disparity, alpha=0,
                     beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 生成深度图（颜色图）
dis_color = disparity
dis_color = cv2.normalize(
    dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
dis_color = cv2.applyColorMap(dis_color, 2)

# 计算三维坐标数据值
threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=1000)
# 计算出的threeD，需要乘以16，才等于现实中的距离
threeD = threeD * 16

cv2.imwrite('depth.jpg', dis_color)
# cv2.imshow("depth", dis_color)
# cv2.imshow("disparity", disp)  # 显示深度图的双目画面
# while True:
#     if cv2.waitKey(1) == ord('q'):
#         break
