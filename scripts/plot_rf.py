import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def load_sparse(stem, num_pre, num_post):
    g = np.load(stem + "-g.npy")
    pre_ind = np.load(stem + "-pre_ind.npy")
    post_ind = np.load(stem + "-post_ind.npy")
    assert len(g) == len(pre_ind)
    assert len(g) == len(post_ind)
    assert np.amax(pre_ind) < num_pre
    assert np.amax(post_ind) < num_post
    data = np.zeros((num_pre, num_post))
    data[pre_ind,post_ind] = g
    return data

num_e = int(round(512 * 0.6))
num_i = 512 - num_e

path = os.path.join("weights", "checkpoints_0_False_0_512_100_dvs_gesture_1_1234_0.1_512_True_alif_0.01_0.05_0.6_full")
ee = load_sparse(os.path.join(path, "99-Conn_InputE_HiddenE0"), 32*32*2, num_e)
ie = load_sparse(os.path.join(path, "99-Conn_InputI_HiddenE0"), 32*32*2, num_e)
e = ee + ie

min_g = np.amin(e)
max_g = np.amax(e)

pal = sns.color_palette("icefire", as_cmap=True)

fig, axes = plt.subplots(13, 24, sharex="col", sharey="row")
for i, a in zip(range(e.shape[1]), axes.flatten()):
    rf = np.reshape(e[:,i], (32,32,2))
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.imshow(rf[:,:,1], vmin=-0.4, vmax=0.4, cmap=pal)
for a in axes.flatten()[e.shape[1]:]:
    a.set_axis_off()
fig.tight_layout(pad=0)
plt.show()