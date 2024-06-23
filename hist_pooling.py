# 将时空特征使用10柱柱状图池化
import os
import numpy as np
import matplotlib.pyplot as plt


def hist_pooling(features, hist_nb=10):
    global_min = features.min()
    global_max = features.max()

    r = features.mean()*2

    bins = np.linspace(global_min, r, num=hist_nb+1)
    histograms = np.zeros((len(features), hist_nb))

    for i, obj_feature in enumerate(features):
        hist, _ = np.histogram(obj_feature, bins=bins)
        histograms[i] = hist
    return histograms

# step 1 : 读取所有的特征
st_features_dir = 'st_features/'
st_features = os.listdir(st_features_dir)
s_features = []
t_features = []
for file in st_features:
    s_f, t_f = np.load(st_features_dir+file)
    s_features.append(s_f)
    t_features.append(t_f)

# step 2 : 找到柱状图边界并为时间、空间特征分别创建10柱状图
s_hist = hist_pooling(np.array(s_features),10)
t_hist = hist_pooling(np.array(t_features),10)

# 用于测试
# for i in range(len(s_hist)):
#     plt.clf()
#     plt.plot(s_hist[i], label = 's_hist')
#     plt.plot(t_hist[i],label = 't_hist')
#     plt.legend()
#     plt.show()

# step 3 : 拼接成20维特征并保存
st_pooled_features = np.hstack((s_hist,t_hist))
# 用于测试
# for i in range(len(st_pooled_features)):
#     plt.clf()
#     plt.plot(s_hist[i], label = 's_hist')
#     plt.plot(t_hist[i],label = 't_hist')
#     plt.plot(st_pooled_features[i])
#     plt.legend()
#     plt.show()
np.save('st_pooled_features/st_pooled_features.npy',st_pooled_features)

