import numpy as np
import tensorflow as tf

class BrainTumorModel:
    def __init__(self):
        self.model_path = 'C:/Users/Ahmed/Documents/GitHub/ai-model-api/models/trained_models/brain_model.h5'
        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

    def predict(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))[0]
        top_two_indices = np.argsort(prediction)[-2:][::-1]
        top_two = [{'class_name': self.class_names[idx], 'percent': f'{prediction[idx]:.2%}'} for idx in top_two_indices]
        return prediction, top_two
