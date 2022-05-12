import base64, fcntl, json, subprocess, time
from random import sample
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import *

app = Flask(__name__)
cors = CORS(app, resources={"*": {"origins": "*"}})

sub = None

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
        mlp_model = MLP_ODE(num_nodes, num_layers, num_output, num_cond)
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
    global sub
    if sub is not None:
        sub.kill()
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
    if sub is None:
        line = ""
    else:
        lines = []
        while True:
            line = sub.stdout.readline().decode()
            lines.append(line)
            if line == '':
                break
    return jsonify({
        "nextLine": '\n'.join(lines),
    })