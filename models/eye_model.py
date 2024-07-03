import numpy as np
import tensorflow as tf
import os
from PIL import Image

class EyeModel:
    def __init__(self):

        rel_path = r"models/trained_models/eye model_mobilenet_93.h5"  # Raw string with forward slashes
        self.model_path = os.path.join(os.getcwd(), rel_path)  # Combine with current working directory

        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names=['cataract','diabetic_retinopathy','glaucoma','normal']


    def predict(self, image):
        predictions = self.model.predict(image)
        
        max_prob_index = np.argmax(predictions)  # Index of class with maximum probability
        max_prob_class = self.class_names[max_prob_index]  # Name of the predicted class
        max_prob_percentage = predictions[0, max_prob_index] * 100  # Percentage probability
        prediction={
            'className':max_prob_class,
            'percentage':f'{max_prob_percentage:.2f}'
        }
        return prediction
