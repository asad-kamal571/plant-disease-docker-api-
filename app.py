from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# Limit TensorFlow GPU memory usage
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Load the trained model once when the server starts
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Full class name list (38)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# 🔧 Helper function for center cropping
def center_crop(image: Image.Image, crop_fraction=0.85) -> Image.Image:
    width, height = image.size
    new_width = int(width * crop_fraction)
    new_height = int(height * crop_fraction)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image field found in request'}), 400

        base64_image = data['image']

        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # 🔧 Apply center crop before resizing
        image = center_crop(image, crop_fraction=0.85)
        image = image.resize((128, 128))

        # ✅ Optional: Save for debugging
        os.makedirs("received_images", exist_ok=True)
        image.save("received_images/cropped_from_expo.jpg")

        # Preprocess image for model (normalize like training)
        image_array = np.array(image) / 255.0
        input_arr = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(input_arr)
        result_index = int(np.argmax(prediction))
        result_class = class_names[result_index]
        confidence = float(np.max(prediction))

        return jsonify({
            'prediction': result_class,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/')
def home():
    return '🌿 Plant Disease Flask API is Running!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

