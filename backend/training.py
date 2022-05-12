import sys
from model import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

args = sys.argv
# assert there are 7 arguments, file exists
_, lr, stacked_ffjords, num_layers, num_nodes, num_output, num_cond, batch_size = args
lr = float(lr)
stacked_ffjords = int(stacked_ffjords)
num_layers = int(num_layers)
num_nodes = int(num_nodes)
num_output = int(num_output)
num_cond = int(num_cond)
batch_size = int(batch_size)
path = "/tmp/samples_train.npy"

stacked_mlps = []
for _ in range(stacked_ffjords):
    mlp_model = MLP_ODE(num_nodes, num_layers, num_output, num_cond)
    stacked_mlps.append(mlp_model)

#Create the model
model = FFJORD(stacked_mlps, batch_size, num_output ,trace_type='hutchinson')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

callbacks=[
        EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=1e-6, verbose=1, cooldown=0)
]

with open(path, 'rb') as f:
    samples_train = np.load(f)

history = model.fit(
    samples_train,
    batch_size=batch_size,
    epochs=40,
    verbose=1,
    validation_split=0.1,
    callbacks=callbacks,
)

save_model(model)