import sys
import matplotlib.pyplot as plt
import math
import numpy as np

#samples_train = sys.argv[1]
#xs = np.linspace(0, 1, 100)
#ys = [eval(y_func) for x in xs]
with open('/tmp/samples_train.npy', 'rb') as f:
    samples_train = np.load(f)
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(samples_train[:,0], samples_train[:,1], color='r')
fig.savefig("/tmp/dataset.png")