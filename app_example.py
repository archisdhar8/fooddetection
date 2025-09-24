"""
Simple App Example for Food Detection Model
This demonstrates how to use the trained model in a web application
"""

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import pickle
from PIL import Image
import io
import base64

app = Flask(__name__)

class FoodDetectionApp:
    def __init__(self, model_path='models/food_detection_model.h5'):
        """
        Initialize the food detection app with the trained model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.image_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model and metadata
        """
        try:
            # Load the model
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            
            # Load metadata
            metadata_path = 'models/model_metadata.pkl'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.class_names = metadata['class_names']
                self.image_size = metadata['image_size']
                print(f"Model metadata loaded. Classes: {self.class_names}")
            else:
                # Fallback class names if metadata not available
                self.class_names = ['freshapples', 'freshbanana', 'freshoranges', 
                                  'rottenapples', 'rottenbanana', 'rottenoranges']
                print("Using default class names")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_file):
        """
        Preprocess uploaded image for prediction
        """
        try:
            # Open and resize image
            image = Image.open(image_file)
            image = image.convert('RGB')
            image = image.resize(self.image_size)
            
            # Convert to array and normalize
            img_array = np.array(image)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_file):
        """
        Predict the class of an uploaded image
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_file)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Determine if fresh or rotten
            is_fresh = 'fresh' in predicted_class.lower()
            fruit_type = predicted_class.replace('fresh', '').replace('rotten', '').replace('apples', 'apple').replace('banana', 'banana').replace('oranges', 'orange')
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_fresh': is_fresh,
                'fruit_type': fruit_type,
                'status': 'Fresh' if is_fresh else 'Rotten'
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {'error': str(e)}

# Initialize the app
food_app = FoodDetectionApp()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Food Detection App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { text-align: center; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; margin: 20px 0; border-radius: 10px; }
        .result { margin: 20px 0; padding: 20px; border-radius: 10px; }
        .fresh { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .rotten { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        input[type="file"] { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍎🍌🍊 Food Detection App</h1>
        <p>Upload an image of fruit to detect if it's fresh or rotten!</p>
        
        <form action="/predict" method="post" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" name="image" accept="image/*" required>
                <br><br>
                <button type="submit">Detect Food Quality</button>
            </div>
        </form>
        
        {% if result %}
            <div class="result {{ 'fresh' if result.is_fresh else 'rotten' if 'predicted_class' in result else 'error' }}">
                {% if 'error' in result %}
                    <h3>❌ Error</h3>
                    <p>{{ result.error }}</p>
                {% else %}
                    <h3>{% if result.is_fresh %}✅ Fresh{% else %}❌ Rotten{% endif %} {{ result.fruit_type.title() }}</h3>
                    <p><strong>Predicted Class:</strong> {{ result.predicted_class }}</p>
                    <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence * 100) }}%</p>
                    <p><strong>Status:</strong> {{ result.status }}</p>
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'})
        
        # Make prediction
        result = food_app.predict(file)
        
        # Render result on the same page
        return render_template_string(HTML_TEMPLATE, result=result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Make prediction
        result = food_app.predict(file)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classes')
def get_classes():
    """Get available classes"""
    return jsonify({'classes': food_app.class_names})

if __name__ == '__main__':
    print("Starting Food Detection App...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
