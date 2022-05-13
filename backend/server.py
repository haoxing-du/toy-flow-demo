import base64, fcntl, json, subprocess, time
from random import sample
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import *

app = Flask(__name__)
cors = CORS(app, resources={"*": {"origins": "*"}})

sub = None
lines = []

@app.route('/get_square', methods=["POST"])
def user():
    data = json.loads(request.data)
    print("I got the following request:", data)
    return jsonify({
        "result": data["value"]**2,
    })

@app.route('/plot_function', methods=["POST"])
def plot_function():
    data = json.loads(request.data)
    y_func = data["y_func"]
    subprocess.check_call(["python", "render.py", y_func])
    with open("/tmp/output.png", "rb") as f:
        png_data = f.read()
    formatted_png = "data:image/png;base64," + base64.b64encode(png_data).decode()
    return jsonify({
        "pngData": formatted_png,
    })

@app.route('/generate_dataset', methods=["POST"])
def generate_dataset():
    data = json.loads(request.data)
    num_batches = data["num_batches"]
    dataset_size = int(num_batches) * 256
    samples_train = generate_2D_gaussian(dataset_size)
    with open('/tmp/samples_train.npy', 'wb') as f:
        np.save(f, samples_train)
    subprocess.check_call(["python", "plot_dataset.py"])
    with open("/tmp/dataset.png", "rb") as f:
        png_data = f.read()
    formatted_png = "data:image/png;base64," + base64.b64encode(png_data).decode()
    return jsonify({
        "pngData": formatted_png,
    })

@app.route('/calculate_num_params', methods=["POST"])
def calculate_num_params():
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

    sub = subprocess.Popen(["/Users/haoxingdu/ay250_env/bin/python", "training.py", str(lr), str(stacked_ffjords),\
        str(num_layers), str(num_nodes), str(num_output), str(num_cond), \
        str(batch_size)], \
        stdout=subprocess.PIPE, env={"PYTHONUNBUFFERED": "true"},
    )

    fl = fcntl.fcntl(sub.stdout, fcntl.F_GETFL)
    fcntl.fcntl(sub.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    #for i in range(100):
    #    print(sub.stdout.readline())
    #    time.sleep(0.5)

    return jsonify({})

@app.route('/get_updates', methods=["POST"])
def get_updates():
    global lines
    if sub is not None:
        while True:
            line = sub.stdout.readline()
            #print(line)
            line = line.decode()
            if line == '':
                break
            list_of_lines = line.split("\r")
            line = list_of_lines[-1]
            if line[0] == 'E' or len(lines) > 0 and lines[-1][0] == 'E':
                lines.append(line)
            else:
                #print(list_of_lines, line)
                lines[-1] = line
    return jsonify({
        "nextLine": ''.join(lines),
    })

@app.route('/plot_samples', methods=["POST"])
def plot_samples():
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