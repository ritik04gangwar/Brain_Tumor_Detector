from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize Flask application
app = Flask(__name__)

# Load the model
model = load_model('my_model.keras')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image file is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Load the image and preprocess
    image = load_img(file, target_size=(224, 224))  # Adjust size based on your model input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)  # Assuming classification task

    # Return result
    return jsonify({'predicted_class': int(predicted_class[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
