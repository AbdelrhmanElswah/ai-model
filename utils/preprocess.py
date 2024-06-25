from PIL import Image
import numpy as np
import cv2

def preprocess_image(file, target_size):
    
    image = Image.open(file.stream).convert('RGB')  # Convert image to RGB
    # Convert the image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    image_resized = cv2.resize(open_cv_image, target_size)
    image_array = np.expand_dims(image_resized, axis=0)  # Expand dimensions to match model input
    
    return image_array
