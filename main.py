# 用于多核并行计算任务分配
# TODO:
# 2. 找到spatial_masking的bug


from multiprocessing import Pool
from cpu_task import task_function
import os

if __name__ == "__main__":
    # 存放所有原视频.264文件的文件夹
    # raw_video_dir = 'C:/Users/Jiawen/Desktop/st_features/'
    # processor_nb = 2
    # # 用于分配任务的视频文件名列表
    # video_list = os.listdir(raw_video_dir)
    # tasks = [[raw_video_dir, filename] for filename in video_list]
    # # 每个处理器分到一个视频完整的特征提取任务
    # with Pool(processor_nb) as pool:
    #     pool.map(task_function, tasks)

    # 存放所有原视频.264文件的文件夹
    raw_video_dir = 'D:/workspace/matlab/JND_prediction/dataset/264videos/'
    # 用于分配任务的视频文件名列表
    video_list = os.listdir(raw_video_dir)
    tasks = [[raw_video_dir, filename] for filename in video_list]
    for e in tasks:
        task_function(e)