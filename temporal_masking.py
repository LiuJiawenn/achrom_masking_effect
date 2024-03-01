# 输入一个视频灰度矩阵，返回时间掩蔽响应（这个视频一起算）

import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
import matplotlib.pyplot as plt
import time


def block_masking(Y,y,patch_h, patch_w):
    # 计算一个patch（例：90*160*30）内的时间掩蔽响应
    # patch需整理 Y=14400*30帧
    # 下面一帧需整理成 y=14400向量
    s_nb = Y.shape[1]
    U, S, Vh = np.linalg.svd(Y, full_matrices=True, compute_uv=True)
    U_n = U[:, :s_nb]
    S_n = np.diag(S[:s_nb])
    Vh_n = Vh[:s_nb, :]

    xkl = np.dot(S_n, Vh_n)
    xl = xkl[:, -1]
    x_kplus1_l = xkl[:, 1:]
    x_k_lminus1 = xkl[:, 0:-1]

    al = x_kplus1_l @ np.linalg.pinv(x_k_lminus1)
    cl = U_n

    RT_l_plus_1 = np.abs(y - cl @ al @ xl)
    RT_l_plus_1 = RT_l_plus_1.reshape(patch_h, patch_w)
    return RT_l_plus_1


def temporal_masking_effect(gray_video_matrics, patch_h, patch_w, patch_d):
    frame_nb = len(gray_video_matrics)
    h = gray_video_matrics.shape[1]
    w = gray_video_matrics.shape[2]

    n = 2*frame_nb//patch_d-2  # 最后返回的响应帧数

    restored_masks = np.zeros((n, h, w))
    # 设置patch大小的窗口，按照行，列, 深度移动
    count = 0
    running_h = 0
    for w_d in range(0, frame_nb, patch_d//2):
        if w_d+patch_d >= frame_nb:  # 最后15帧舍掉不要
            break

        print('正在计算第' + str(count) + '帧')
        start_time = time.time()
        for w_h in range(0, h, patch_h):
            running_h += 1
            print('h='+str(running_h))
            for w_w in range(0, w, patch_w):
                # 窗口内整理成Y， 窗口后面的一帧整理成y
                Y_raw = gray_video_matrics[w_d:w_d+patch_d, w_h:w_h+patch_h, w_w:w_w+patch_w]  # 确定窗口位置
                Y = np.transpose(Y_raw,(1,2,0)).reshape(patch_w*patch_h,patch_d)  # 调换维度顺序后，合并前两个维度
                y = gray_video_matrics[w_d+patch_d, w_h:w_h+patch_h, w_w:w_w+patch_w].reshape(patch_w*patch_h)

                # 根据Y，y计算时间掩蔽
                block_mask = block_masking(Y, y, patch_h, patch_w)
                # 暂存掩蔽响应
                restored_masks[count, w_h:w_h+patch_h, w_w:w_w+patch_w] = block_mask
        end_time = time.time()
        print(f"运行时间：{end_time - start_time}秒")
        count += 1
        # 下面用于测试
        # x = restored_masks[count-1]
        # cm2 = plt.cm.get_cmap('jet')
        # plt.imshow(x, cmap = cm2)
        # plt.colorbar()
        # plt.show()
        # print('aa')

    # 返回响应矩阵
    return restored_masks

#  下面用于测试函数 temporal_masking_effect
gray_video_matrics = np.load('data/gray_frames.npy')
mask = temporal_masking_effect(gray_video_matrics, patch_h=90, patch_w=160, patch_d=30)
np.save('data/tR.py',mask)
x = mask[0]
cm2 = plt.cm.get_cmap('jet')
plt.imshow(x,vmax = 1000, cmap = cm2)
plt.colorbar()
# # plt.savefig('spatialMasking5.png', dpi=300)
plt.show()





