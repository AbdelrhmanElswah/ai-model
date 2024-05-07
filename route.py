import tensorflow as tf
from PIL import Image
import numpy as np

model_path = 'D:/Projects/mobilenetv2_1.00_224-Covid-19-94.87_RMSProp.h5'
model = tf.keras.models.load_model(model_path)


class_names = ["Class 1", "Class 2", "Class 3", "Class 4"]

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  
    return image

image_path = 'D:/Projects/flaskk/image(1).jpg' 
image = Image.open(image_path)
processed_image = preprocess_image(image)

prediction = model.predict(np.expand_dims(processed_image, axis=0))

# Zip class names with probabilities and print
for class_name, prob in zip(class_names, prediction[0]):
    print(f"{class_name}: {prob}")






