from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load your trained model
model_path = 'C:/Users/Ahmed/Documents/GitHub/ai-model-api/models/Covid-19.h5'
model = tf.keras.models.load_model(model_path)
class_names = ["Class 1", "Class 2", "Class 3", "Class 4"]

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        # Read the image via file.stream
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        
        # Prepare the response
        response = {class_name: float(prob) for class_name, prob in zip(class_names, prediction[0])}
        
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
