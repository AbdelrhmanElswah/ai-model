from PIL import Image
import numpy as np

def preprocess_image(file, target_size):
    image = Image.open(file.stream).convert('RGB')  # Convert image to RGB
    image = image.resize(target_size)  # Resize image
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize image if needed (assuming model was trained with normalization)
    return image_array
