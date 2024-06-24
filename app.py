from flask import Flask, request, jsonify
from models.brain_tumor_model import BrainTumorModel
from models.covid19_model import Covid19Model
from utils.preprocess import preprocess_image
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Initialize your models
brain_tumor_model = BrainTumorModel()
covid19_model = Covid19Model()

@app.route('/predict/brain_tumor', methods=['POST'])
def predict_brain_tumor():
    if 'file' not in request.files:
        return jsonify({'code': 400, 'status': 'error', 'data': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'code': 400, 'status': 'error', 'data': 'No file selected'}), 400
    if file:
        image = preprocess_image(file, target_size=(150, 150))
        prediction, top_two = brain_tumor_model.predict(image)
        return jsonify({
            'code': 200,
            'status': 'success',
            'data': {
                '1st_prediction': top_two[0],
                '2nd_prediction': top_two[1]
            }
        }), 200

@app.route('/predict/covid19', methods=['POST'])
def predict_covid19():
    if 'file' not in request.files:
        return jsonify({'code': 400, 'status': 'error', 'data': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'code': 400, 'status': 'error', 'data': 'No file selected'}), 400
    if file:
        file_path = request.files['file'].stream  # Use file stream for direct access
        prediction, top_two = covid19_model.predict(file_path)
        print(prediction)
        print(top_two)

        return jsonify({
            'code': 200,
            'status': 'success',
            'data': {
                '1st_prediction': top_two[0],
                '2nd_prediction': top_two[1]
            }
        }), 200

if __name__ == '__main__':
    app.run(debug=True)
