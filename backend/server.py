import json, base64, subprocess
from random import sample
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import *

app = Flask(__name__)
cors = CORS(app, resources={"*": {"origins": "*"}})

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