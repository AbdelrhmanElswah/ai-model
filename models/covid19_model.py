import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

class Covid19Model:
    def __init__(self):
        self.model_path = 'C:/Users/Ahmed/Documents/GitHub/ai-model-api/models/trained_models/Covid-19.h5'
        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

    def preprocess_image(self, image_path):
        # Load the image using PIL and convert to RGB
        pil_image = Image.open(image_path).convert('RGB')
        # Convert the image to OpenCV format
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
        # Resize the image to the target size
        image_resized = cv2.resize(open_cv_image, (224, 224))
        # Normalize the image if needed
        image_array = image_resized / 255.0
        # Expand dimensions to match model input
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        prediction = self.model.predict(image)[0]
        top_two_indices = np.argsort(prediction)[-2:][::-1]
        top_two = [{'class_name': self.class_names[idx], 'percent': f'{prediction[idx]:.2%}'} for idx in top_two_indices]
        return prediction, top_two
