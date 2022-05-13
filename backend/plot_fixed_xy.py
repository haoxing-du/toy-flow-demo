import sys
import matplotlib.pyplot as plt
import numpy as np
from toy_flow_model import *

args = sys.argv
_, x, y, stacked_ffjords, num_layers, num_nodes, num_output, num_cond, batch_size = args
x, y = float(x), float(y)
stacked_ffjords = int(stacked_ffjords)
num_layers = int(num_layers)
num_nodes = int(num_nodes)
num_output = int(num_output)
num_cond = int(num_cond)
batch_size = int(batch_size)

stacked_mlps = []
for _ in range(stacked_ffjords):
    mlp_model = MLP_ODE(num_nodes*num_output, num_layers, num_output, num_cond)
    stacked_mlps.append(mlp_model)
model = FFJORD(stacked_mlps, batch_size, num_output, \
    trace_type='exact', name='loaded_model')
load_model(model)

a_array = np.linspace(0,1,101).reshape((101,1)).astype(np.float32)
points = np.full((101,2), (x,y)).astype(np.float32)
probs = model.conditional_prob(points, a_array)
fig = plt.figure(figsize=(6,3))
plt.plot(a_array, probs, label="learned", c='orange')

analytical = np.array([np.exp(-(x-a)**2/2) * np.exp(-(y-a)**2/2)/(2*np.pi) for a in a_array]).reshape((101,))
plt.plot(a_array, analytical, label="analytical")
plt.xlabel("a")
plt.legend()
plt.tight_layout()
fig.savefig("/tmp/fixed_xy.png")