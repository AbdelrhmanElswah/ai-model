import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

class Covid19Model:
    def __init__(self):
        self.model_path = r"E:\Mazen\هندسة\Graduation_Project\my_work\ai-model-api-main\models\trained_models\chest_model_mobilenet_95.h5"
        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']



    def predict(self, image):
        predictions = self.model.predict(image)
        
        max_prob_index = np.argmax(predictions)  # Index of class with maximum probability
        max_prob_class = self.class_names[max_prob_index]  # Name of the predicted class
        max_prob_percentage = predictions[0, max_prob_index] * 100  # Percentage probability
        prediction_text = f'Predicted Class: {max_prob_class} ({max_prob_percentage:.2f}%)'
        
        return prediction_text
