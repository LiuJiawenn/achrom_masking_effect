# 输入一帧，返回该帧的空间掩蔽响应

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def neighbour_list(block=9):
    # 返回邻域像素与中心像素的相对位置
    # 返回的是圆形扩张的稀疏的邻域像素
    if block % 2 == 0:
        print("block size should be single!")
        return
    res = []
    order = (block-1)//2
    for i in range(1, order):
        if i % 2 == 1:
            res += [(-i, 0), (0, i), (i, 0), (0, -i)]
        else:
            res += [(-i, -i), (-i, i), (i, i), (i, -i)]
    return res


def bolck_masking(X, Y):
    # 用于计算一个block （例如，9×9范围内）的空间掩蔽
    N = X.shape[0]-1
    S = np.zeros(N+1)

    RXP = np.linalg.pinv(X.T@X / N)
    RYX = Y.T@X / N

    # print(temp.shape)
    for i in range(N+1):
        S[i] = np.abs(Y[i]-RYX@RXP@X[i].T)

    l = int(np.sqrt(N+1))
    S = S.reshape(l,l)
    return S


# 返回blocks的起始坐标
def blockindex(r, c, block):
    # 用于返回所有block的起始坐标
    res = []
    for i in range(0, r, block):
        for j in range(0, c, block):
            if i + block <= r and j + block <= c:
                res.append([i, j])
    return res


def get_neighborhood(image, neighbour_offsets):
    # 返回每个像素的邻域像素值列表
    # 例如使用 9*9 block 时，每个像素有12个邻域像素点
    # 本函数会返回一个 1080*1920*12 的矩阵
    rows, cols = image.shape
    pad_width = max(max(abs(dx), abs(dy)) for dx, dy in neighbour_offsets)
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    neighborhood = np.empty((rows, cols, len(neighbour_offsets)), dtype=image.dtype)

    for i, (dx, dy) in enumerate(neighbour_offsets):
        neighborhood[:, :, i] = padded_image[pad_width + dx:pad_width + dx + rows,
                                             pad_width + dy:pad_width + dy + cols]

    return neighborhood


def spatial_masking_effect(gray, block=9):
    # 本函数将返回gray的空间掩蔽响应
    # img 输入为帧矩阵，例如1080×1920的 ndarray
    start_time = time.time()
    # gray = gray / 255.0  之前处理原视频需要转换为0-1区间。现在处理亮度矩阵已经转换好了
    neighbourList = neighbour_list(block)
    neighborhood_pixels = get_neighborhood(gray, neighbourList)

    r = gray.shape[0]
    c = gray.shape[1]
    blocks = blockindex(r, c, block)
    masks = []

    for x, y in blocks:
        Y = gray[x:x+block,y:y+block].reshape(-1, 1)
        X = neighborhood_pixels[x:x+block,y:y+block].reshape(-1, neighborhood_pixels.shape[2])

        S = bolck_masking(X, Y)
        masks.append(S)
    #
    rows = []

    r = 1080 // block
    c = 1920 // block
    for i in range(r):
        # 每一行有 6 个图像块，水平堆叠
        row = np.hstack(masks[i * c:(i + 1) * c])
        rows.append(row)

    # 将所有行垂直堆叠起来形成最终的图像
    final_image = np.vstack(rows)
    # 因为1920除不开block = 9 因此把最后剩下的三列补0
    res = np.pad(final_image, ((0,0),(0,3)),'constant', constant_values=(0,))
    end_time = time.time()
    print(f"运行时间：{end_time - start_time}秒")
    return res

# 下面用于测试函数 spatial_masking_effect
# img = cv2.imread('C:/Users/Jiawen/Desktop/videoPreview/videoSRC142_1920x1080_24.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# f = spatial_masking_effect(gray, block=9)
# plt.imshow(f)
# plt.colorbar()
# plt.show()