# 输入一个视频灰度矩阵，返回时间掩蔽响应（这个视频一起算）
import cupy as cp
import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
import matplotlib.pyplot as plt
import time

def extract_patches(arr, patch_shape=(32, 32, 3), extraction_step=(32, 32, 3)):
    # input (1080,1920,3)
    # 对应维度+1需要跳过的字节数
    patch_strides = arr.strides
    # 四个维度，每个维度创建一个切片
    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    # patch_indices_shape = (10,33,60,1)
    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    patches = patches.reshape((-1, patch_shape[0]* patch_shape[1]))
    return patches

def block_masking(Y,y,patch_h, patch_w):
    # 计算一个patch（例：90*160*30）内的时间掩蔽响应
    # patch需整理 Y=14400*30帧
    # 下面一帧需整理成 y=14400向量
    s_nb = Y.shape[1]
    U, S, Vh = cp.linalg.svd(Y, full_matrices=True, compute_uv=True)
    U_n = U[:, :s_nb]
    S_n = cp.diag(S[:s_nb])
    Vh_n = Vh[:s_nb, :]

    xkl = cp.dot(S_n, Vh_n)
    xl = xkl[:, -1]
    x_kplus1_l = xkl[:, 1:]
    x_k_lminus1 = xkl[:, 0:-1]

    al =cp.dot(x_kplus1_l, cp.linalg.pinv(x_k_lminus1))
    cl = U_n

    RT_l_plus_1 = cp.abs(y - cl @ al @ xl)
    RT_l_plus_1 = RT_l_plus_1.reshape(patch_h, patch_w)
    return RT_l_plus_1


def temporal_masking_effect(gray_video_matrics, patch_h, patch_w, patch_d):
    frame_nb = len(gray_video_matrics)
    h = gray_video_matrics.shape[1]
    w = gray_video_matrics.shape[2]

    n = 2*frame_nb//patch_d-2  # 最后返回的响应帧数
    gray_video_matrics = cp.array(gray_video_matrics)
    restored_masks = cp.zeros((n, h, w))
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
                Y_raw = cp.array(gray_video_matrics[w_d:w_d+patch_d, w_h:w_h+patch_h, w_w:w_w+patch_w]) # 确定窗口位置
                Y = cp.transpose(Y_raw,(1,2,0)).reshape(patch_w*patch_h,patch_d)  # 调换维度顺序后，合并前两个维度
                y =cp.array(gray_video_matrics[w_d+patch_d, w_h:w_h+patch_h, w_w:w_w+patch_w]).reshape(patch_w*patch_h)

                # 根据Y，y计算时间掩蔽
                block_mask = block_masking(Y, y, patch_h, patch_w)
                # 暂存掩蔽响应
                restored_masks[count, w_h:w_h+patch_h, w_w:w_w+patch_w] = block_mask
        end_time = time.time()
        print(f"运行时间：{end_time - start_time}秒")
        count += 1

    # 返回响应矩阵
    return restored_masks

# 下面用于测试函数 temporal_masking_effect
gray_video_matrics = np.load('data/gray_frames.npy')
mask = temporal_masking_effect(gray_video_matrics, patch_h=90, patch_w=160, patch_d=30)
np.save('data/tr_test.py',mask)
x = mask[0]
cm2 = plt.cm.get_cmap('jet')
plt.imshow(x,vmax = 1000, cmap = cm2)
plt.colorbar()
# # plt.savefig('spatialMasking5.png', dpi=300)
plt.show()


# mask = np.load('data/tR_test.npy')
# for i in range(len(mask)):
#     x = mask[i]
#     cm2 = plt.cm.get_cmap('jet')
#     plt.imshow(x,vmax = 1000, cmap = cm2)
#     plt.colorbar()
#     # # plt.savefig('spatialMasking5.png', dpi=300)
#     plt.show()