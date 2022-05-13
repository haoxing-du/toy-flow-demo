import sys
import matplotlib.pyplot as plt
import numpy as np
from toy_flow_model import *

args = sys.argv
_, a, stacked_ffjords, num_layers, num_nodes, num_output, num_cond, batch_size = args
a = float(a)
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

xs = np.linspace(-4,4,101)
xs_2D = np.stack([xs, xs]).T.astype(np.float32)
a_array = np.ones((101,1), dtype=np.float32) * a
probs = model.conditional_prob(xs_2D, a_array)
fig = plt.figure(figsize=(6,3))
plt.plot(xs, probs, label="learned", c='orange')

analytical = np.array([np.exp(-(x-a)**2)/(2*np.pi) for x in xs]).reshape((101,))
plt.plot(xs, analytical, label="analytical")
plt.xlabel("x")
plt.legend()
plt.tight_layout()
fig.savefig("/tmp/fixed_a.png")