import matplotlib.pyplot as plt
import numpy as np
from os.path import exists

path_transformed = "/tmp/transformed.npy"
# assert file exists
with open("/tmp/transformed.npy", 'rb') as f:
    transformed = np.load(f)
with open("/tmp/transformed_first.npy", 'rb') as f:
    transformed_first = np.load(f)
with open("/tmp/transformed_second.npy", 'rb') as f:
    transformed_second = np.load(f)

fig = plt.figure(figsize=(12, 5))
plt.subplot(121)    
plt.scatter(transformed[:, 0], transformed[:, 1], color="r")
plt.subplot(122)
plt.scatter(transformed_first[:, 0], transformed_first[:, 1], color="g")
plt.scatter(transformed_second[:, 0], transformed_second[:, 1], color="b")

fig.savefig("/tmp/samples.png")