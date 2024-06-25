import numpy as np
import tensorflow as tf

class BrainTumorModel:
    def __init__(self):
        self.model_path = r"C:\xampp\htdocs\dashboard\ai-model-api\models\trained_models\brain_model_mobilenet_98.h5"
        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

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
