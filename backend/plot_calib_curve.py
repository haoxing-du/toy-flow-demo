import sys
import matplotlib.pyplot as plt
import numpy as np
from toy_flow_model import *

args = sys.argv
_, n_points, stacked_ffjords, num_layers, num_nodes, num_output, num_cond, batch_size = args
n_points = int(n_points)
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

def mle_math(x):
    a, b = x
    if (a+b)/2 < 0:
        return 0
    elif (a+b)/2 > 1:
        return 1
    else:
        return (a+b)/2

def scanner(point, N):
    points = np.array([point] * (N+1))
    zs = np.array([i/N for i in range(N+1)]).reshape((N+1,1))
    log_probs = model.conditional_log_prob(points, zs)
    max_index = np.argmax(log_probs)
    c_max = max_index / N
    
    return c_max

N = 250
points = np.random.uniform(-0.5, 1.5, (n_points,2)).astype(np.float32)

c_maxs = []
c_max_maths = []

for point in points:
    c_max = scanner(point, N)
    c_max_math = mle_math(point)
    c_maxs.append(c_max)
    c_max_maths.append(c_max_math)

fig = plt.figure(figsize=(5,5))
plt.scatter(c_max_maths, c_maxs)
plt.xlabel("mathematical result")
plt.ylabel("inferred result")
plt.plot([0,0.5,1], [0,0.5,1], c='red')
plt.tight_layout()
fig.savefig("/tmp/calib_curve.png")