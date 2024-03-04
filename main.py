from gpu_task import task_function
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mat_path = ''
    # 用于分配任务的视频文件名列表
    video_list = os.listdir(mat_path)
    tasks = [[mat_path, filename] for filename in video_list]
    for e in tasks:
        task_function(e)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
