from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load your trained model
model_path = 'C:/Users/Ahmed/Documents/GitHub/ai-model-api/models/Covid-19.h5'
model = tf.keras.models.load_model(model_path)
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'code': 400, 'status': 'error', 'data': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'code': 400, 'status': 'error', 'data': 'No file selected'}), 400
    if file:
        # Read the image via file.stream
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]

        # Get indices of the two highest probabilities
        top_two_indices = np.argsort(prediction)[-2:]  # Get the indices of the top two predictions
        top_two_indices = top_two_indices[::-1]  # Reverse to make highest first

        # Prepare the top two predictions data
        top1_class = class_names[top_two_indices[0]]
        top1_prob = prediction[top_two_indices[0]]
        top2_class = class_names[top_two_indices[1]]
        top2_prob = prediction[top_two_indices[1]]

        # Format response
        response = {
            'code': 200,
            'status': 'success',
            'data': {
                'Highest Prediction': {
                    'class_name':top1_class,
                    'percent':f'{top1_prob:.2%}'


            },
                'Second Highest Prediction': {
                    'class_name':top2_class,
                    'percent':f'{top2_prob:.2%}'


            }
            }
        }

        return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
