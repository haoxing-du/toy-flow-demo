import matplotlib.pyplot as plt
import numpy as np
from os.path import exists

path = "/tmp/samples_train.npy"
# assert file exists
with open(path, 'rb') as f:
    samples_train = np.load(f)
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(samples_train[:,0], samples_train[:,1], color='r')
fig.savefig("/tmp/dataset.png")