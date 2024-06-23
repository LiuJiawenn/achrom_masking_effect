import numpy as np
import matplotlib.pyplot as plt
sr = np.load('data/tr_test.npy')
# sr = np.load('data/sr_teat.py.npy')
for i in range(len(sr)):
    cm2 = plt.cm.get_cmap('jet')
    plt.imshow(sr[i], vmax = 0.3, cmap=cm2)
    plt.colorbar(fraction = 0.026, pad = 0.03)
    plt.savefig('temporalMasking.png', dpi=300)
    plt.show()