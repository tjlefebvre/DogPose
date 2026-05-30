import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import run_inference

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image field"}), 400
    image_file = request.files["image"]
    return jsonify(run_inference(image_file))

if __name__ == "__main__":
    app.run(port=5000, debug=False)
