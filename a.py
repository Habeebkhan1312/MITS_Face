from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)

# Load your Keras model (.h5)
model = tf.keras.models.load_model('unique_face_expression_model_.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to preprocess the image before prediction
def preprocess_image(image):
    image = image.resize((48, 48))  # Resize image to match the model input size
    image = image.convert('L')  # Convert image to grayscale (if required by the model)
    image = np.array(image) / 255.0  # Normalize image to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.json:
        # Handle real-time video frame (Base64 string)
        image_data = request.json['image'].split(",")[1]  # Extract base64-encoded image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    else:
        # Handle uploaded image (from file input)
        image_file = request.files['image']
        image = Image.open(image_file)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction using the Keras model
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    # Optionally return bounding box data (if faces detected)
    result = {
        'prediction': predicted_label,
        'bbox': {'x': 50, 'y': 50, 'w': 100, 'h': 100}  # Example static bounding box
    }

    return jsonify(result)

if __name__ == '__main__':   
    app.run(debug=True)