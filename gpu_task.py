# 分配给单一cpu的独立的任务

# 本脚本将VideoSet 220个无损视频 转化为 11×11×9 的一维特征向量
# 时间空间特征各 输出一个特征向量

import mat73
import numpy as np
from spatial_masking import spatial_masking_effect
from temporal_masking import temporal_masking_effect
from numpy.lib.stride_tricks import as_strided


# def h264_to_avi(path_264, path_avi):
#     if os.path.exists(path_avi):
#         return
#
#     stream = ffmpeg.input(path_264)
#     stream = ffmpeg.output(stream, path_avi, vcodec='rawvideo')
#     ffmpeg.run(stream)
def read_mat(path,name):
    data = mat73.loadmat(path)
    x = data[name]
    return x

def extract_patches(arr, patch_shape=(1, 180, 320), extraction_step=(1, 90, 160)):
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
    patches = patches.reshape((-1, patch_shape[1], patch_shape[2]))
    return patches


def task_function(task_id):
    raw_video_dir = task_id[0]
    video = task_id[1]
    print("正在处理视频：", video)

    # path
    feature_dir = 'st_features/'  # 用于保存所有计算好的特征

    # config:
    #    空间掩蔽需要:
    block = 9
    #    时间掩蔽需要:
    patch_h = 90
    patch_w = 160
    patch_d = int(video.split('_')[2])  # 与帧率保持一致，30或者24

    # step 1： 获取mat矩阵
    s_mat = raw_video_dir + video  # 264文件路径
    frames = read_mat(s_mat, 'feature')

    # step 2: 计算空间掩蔽响应
    spatial_masking_frames = []  # 用于保存空间掩蔽响应
    for i in range(len(frames)):
        if i > (patch_d//2) and i % (patch_d//2) == 0:
            spatial_masking_frames.append(spatial_masking_effect(frames[i], block))

    # step 3: 使用frames 矩阵计算时间掩蔽效应矩阵
    print("正在计算" + video + " 时间掩蔽")
    temporal_masking_frames = temporal_masking_effect(np.array(frames), patch_h, patch_w, patch_d)

    # step 4: 将时空特征响应矩阵分成180*320*30的,每个patch计算均值，形成11×11×8 的一维特征向量
    s_patches = extract_patches(np.array(spatial_masking_frames))
    t_patches = extract_patches(np.array(temporal_masking_frames))

    s_feature = np.mean(np.array(s_patches), axis=(1, 2))  # 11×11×8
    t_feature = np.mean(np.array(t_patches), axis=(1, 2))

    # step 5：保存向量
    feature_path = feature_dir + video.split('.')[0] + '.npy'
    # 跑一个存一个，这样如果中间断了，之前跑过的视频还能留着
    np.save(feature_path, [s_feature, t_feature])
