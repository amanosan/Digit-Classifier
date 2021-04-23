from flask import Flask, request, jsonify
from utilities.torch_utils import transform_image, get_prediction

ALLOWED_EXT = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')

        if file is None or file.filename == "":
            return jsonify({'Error': 'No File'})

        if not allowed_file(file.filename):
            return jsonify({'Error': 'Format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {
                'prediction': prediction.item(),
                'class_name': str(prediction.item())
            }

            return jsonify(data)
        except:
            return jsonify({'Error': 'Error during prediction'})
