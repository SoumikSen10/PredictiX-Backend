import os
import sys
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import tensorflow as tf
import contextlib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU devices
tf.config.set_visible_devices([], 'GPU')  # Explicitly tell TensorFlow to not use GPU

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.get_logger().setLevel('ERROR')

# Assuming the model file is in the same directory as this script
model_path = os.path.join(os.path.dirname(__file__), 'bcd_model.h5')

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")
    sys.exit(1)

IMAGE_SIZE = (150, 150)  # Match the size used during model training

def load_and_preprocess_image(img_path, target_size):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_image_class(model, img_path, target_size):
    img = load_and_preprocess_image(img_path, target_size)
    if img is None:
        return "Error processing image"
    
    try:
        predictions = model.predict(img)
        predicted_class = (predictions[0] > 0.5).astype("int32")
        class_labels = ['benign', 'malignant']
        predicted_label = class_labels[predicted_class[0]]
        return predicted_label
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        sys.exit(1)

    predicted_label = predict_image_class(model, img_path, IMAGE_SIZE)
    print(predicted_label)
