import base64, fcntl, json, subprocess, time
from random import sample
from flask import Flask, request, jsonify
from flask_cors import CORS
from toy_flow_model import *

app = Flask(__name__)
cors = CORS(app, resources={"*": {"origins": "*"}})

# global variables that will be used for the terminal emulator
sub = None
lines = []

@app.route('/generate_dataset', methods=["POST"])
def generate_dataset():
    """ Generate and plot a dataset of conditional 2D Gaussians.
    """
    data = json.loads(request.data)
    num_batches = data["num_batches"]
    dataset_size = int(num_batches) * 256
    samples_train = generate_2D_gaussian(dataset_size)

    # store samples_train as file so that plotting script can read it
    with open('/tmp/samples_train.npy', 'wb') as f:
        np.save(f, samples_train)
    
    # direct function call gives scary error and makes Python crash,
    # so use subprocess instead
    subprocess.check_call(["python", "plot_dataset.py"])
    
    with open("/tmp/dataset.png", "rb") as f:
        png_data = f.read()
    # generated image is passed as base64 string
    formatted_png = "data:image/png;base64," + base64.b64encode(png_data).decode()
    return jsonify({
        "pngData": formatted_png,
    })

@app.route('/calculate_num_params', methods=["POST"])
def calculate_num_params():
    """ Calculate the number of parameters in the specified 
        FFJORD model.
    """
    data = json.loads(request.data)
    lr = float(data["lr"])
    stacked_ffjords = int(data["stacked_ffjords"])
    num_layers = int(data["num_layers"])
    num_nodes = int(data["num_nodes"])
    num_output = 2
    num_cond = 1
    batch_size = 256

    stacked_mlps = []
    for _ in range(stacked_ffjords):
        mlp_model = MLP_ODE(num_nodes*num_output, num_layers, num_output, num_cond)
        stacked_mlps.append(mlp_model)

    #Create the model
    model = FFJORD(stacked_mlps, batch_size, num_output ,trace_type='hutchinson')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

    trainable_params = np.sum([np.prod(v.get_shape()) \
        for v in model.trainable_weights])
    nontrainable_params = np.sum([np.prod(v.get_shape()) \
        for v in model.non_trainable_weights])
    total_params = trainable_params + nontrainable_params
    
    return jsonify({
        "numParams": total_params,
    })

@app.route('/train_model', methods=["POST"])
def train_model():
    """ Train the FFJORD as a subprocess, and get the training
        process updated printed out by Keras.
    """
    global sub, lines
    if sub is not None:
        sub.kill()
    lines = []
    data = json.loads(request.data)
    lr = data["lr"]
    stacked_ffjords = data["stacked_ffjords"]
    num_layers = data["num_layers"]
    num_nodes = data["num_nodes"]
    num_output = 2
    num_cond = 1
    batch_size = 256

    # this won't work with regular "python" for some reason
    sub = subprocess.Popen(["/Users/haoxingdu/ay250_env/bin/python", "training.py", str(lr), str(stacked_ffjords),\
        str(num_layers), str(num_nodes), str(num_output), str(num_cond), \
        str(batch_size)], \
        stdout=subprocess.PIPE, env={"PYTHONUNBUFFERED": "true"},
    )

    # to make a terminal emulator, set stdout to be nonblocking
    fl = fcntl.fcntl(sub.stdout, fcntl.F_GETFL)
    fcntl.fcntl(sub.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    return jsonify({})

@app.route('/get_updates', methods=["POST"])
def get_updates():
    """ The frontend calls this periodically to get new lines from
        the stdout of the training progress.
    """
    global lines
    if sub is not None:
        while True:
            line = sub.stdout.readline()
            line = line.decode()
            if line == '':
                break
            # Keras output flushes training progress line by \r
            list_of_lines = line.split("\r")
            line = list_of_lines[-1]
            # This is probably not very robust, but basically,
            # replace the previous line if it is not "Epoch x"
            if line[0] == 'E' or len(lines) > 0 and lines[-1][0] == 'E':
                lines.append(line)
            else:
                lines[-1] = line
    return jsonify({
        "nextLine": ''.join(lines),
    })

@app.route('/plot_samples', methods=["POST"])
def plot_samples():
    """ Take samples from trained model and plot them.
    """
    data = json.loads(request.data)
    num_batches = int(data["num_batches"])
    stacked_ffjords = int(data["stacked_ffjords"])
    num_layers = int(data["num_layers"])
    num_nodes = int(data["num_nodes"])
    num_output = 2
    num_cond = 1
    batch_size = 256
    dataset_size = num_batches * batch_size
    stacked_mlps = []
    for _ in range(stacked_ffjords):
        mlp_model = MLP_ODE(num_nodes*num_output, num_layers, num_output, num_cond)
        stacked_mlps.append(mlp_model)
    model = FFJORD(stacked_mlps, batch_size, num_output, \
        trace_type='exact', name='loaded_model')
    load_model(model)
    
    path = "/tmp/samples_train.npy"
    with open(path, 'rb') as f:
        samples_train = np.load(f)

    transformed = model.flow.sample(
        dataset_size,
        bijector_kwargs=make_bijector_kwargs(
            model.flow.bijector, {'bijector.': {'conditional_input': samples_train[:,2].reshape(dataset_size,1)}})
    )

    transformed_first = model.flow.sample(
        dataset_size,
        bijector_kwargs=make_bijector_kwargs(
            model.flow.bijector, {'bijector.': {'conditional_input': np.ones((dataset_size,1),dtype=np.float32)}})
    )

    transformed_second = model.flow.sample(
        dataset_size,
        bijector_kwargs=make_bijector_kwargs(
            model.flow.bijector, {'bijector.': {'conditional_input': np.zeros((dataset_size,1),dtype=np.float32)}})
    )

    with open('/tmp/transformed.npy', 'wb') as f:
        np.save(f, transformed)
    with open('/tmp/transformed_first.npy', 'wb') as f:
        np.save(f, transformed_first)
    with open('/tmp/transformed_second.npy', 'wb') as f:
        np.save(f, transformed_second)

    subprocess.check_call(["python", "plot_samples.py"])

    with open("/tmp/samples.png", "rb") as f:
        png_data = f.read()
    formatted_png = "data:image/png;base64," + base64.b64encode(png_data).decode()
    return jsonify({
        "pngData": formatted_png,
    })

@app.route('/plot_fixed_a', methods=['POST'])
def plot_fixed_a():
    """ Plot density as a function of x for a fixed a.
    """
    data = json.loads(request.data)
    fixed_a = float(data["fixed_a"])
    stacked_ffjords = int(data["stacked_ffjords"])
    num_layers = int(data["num_layers"])
    num_nodes = int(data["num_nodes"])
    num_output = 2
    num_cond = 1
    batch_size = 256
    stacked_mlps = []
    for _ in range(stacked_ffjords):
        mlp_model = MLP_ODE(num_nodes*num_output, num_layers, num_output, num_cond)
        stacked_mlps.append(mlp_model)
    model = FFJORD(stacked_mlps, batch_size, num_output, \
        trace_type='exact', name='loaded_model')
    load_model(model)

    subprocess.check_call(["python", "plot_fixed_a.py", str(fixed_a), str(stacked_ffjords),\
        str(num_layers), str(num_nodes), str(num_output), str(num_cond), \
        str(batch_size)])
    
    with open("/tmp/fixed_a.png", "rb") as f:
        png_data = f.read()
    formatted_png = "data:image/png;base64," + base64.b64encode(png_data).decode()
    return jsonify({
        "pngData": formatted_png,
    })

@app.route('/plot_fixed_xy', methods=['POST'])
def plot_fixed_xy():
    """ Plot density as a function of a for a fixed point (x,y).
    """
    data = json.loads(request.data)
    fixed_x = float(data["fixed_x"])
    fixed_y = float(data["fixed_y"])
    stacked_ffjords = int(data["stacked_ffjords"])
    num_layers = int(data["num_layers"])
    num_nodes = int(data["num_nodes"])
    num_output = 2
    num_cond = 1
    batch_size = 256
    stacked_mlps = []
    for _ in range(stacked_ffjords):
        mlp_model = MLP_ODE(num_nodes*num_output, num_layers, num_output, num_cond)
        stacked_mlps.append(mlp_model)
    model = FFJORD(stacked_mlps, batch_size, num_output, \
        trace_type='exact', name='loaded_model')
    load_model(model)

    subprocess.check_call(["python", "plot_fixed_xy.py", str(fixed_x), str(fixed_y), \
        str(stacked_ffjords), str(num_layers), str(num_nodes), str(num_output), \
        str(num_cond), str(batch_size)])
    
    with open("/tmp/fixed_xy.png", "rb") as f:
        png_data = f.read()
    formatted_png = "data:image/png;base64," + base64.b64encode(png_data).decode()
    return jsonify({
        "pngData": formatted_png,
    })

@app.route('/calibration_curve', methods=['POST'])
def calibration_curve():
    """ Plot the calibration curve, learned vs actual most likely
        conditional value for some number of random points.
    """
    data = json.loads(request.data)
    n_points = int(data["n_points"])
    stacked_ffjords = int(data["stacked_ffjords"])
    num_layers = int(data["num_layers"])
    num_nodes = int(data["num_nodes"])
    num_output = 2
    num_cond = 1
    batch_size = 256
    stacked_mlps = []
    for _ in range(stacked_ffjords):
        mlp_model = MLP_ODE(num_nodes*num_output, num_layers, num_output, num_cond)
        stacked_mlps.append(mlp_model)
    model = FFJORD(stacked_mlps, batch_size, num_output, \
        trace_type='exact', name='loaded_model')
    load_model(model)

    subprocess.check_call(["python", "plot_calib_curve.py", str(n_points), \
        str(stacked_ffjords), str(num_layers), str(num_nodes), str(num_output), \
        str(num_cond), str(batch_size)])

    with open("/tmp/calib_curve.png", "rb") as f:
        png_data = f.read()
    formatted_png = "data:image/png;base64," + base64.b64encode(png_data).decode()
    return jsonify({
        "pngData": formatted_png,
    })