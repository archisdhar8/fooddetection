"""
Simple Prediction Script for Food Detection Model
Use this script to predict on individual images
"""

import os
import sys
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import argparse

class FoodPredictor:
    def __init__(self, model_path='models/food_detection_model.h5'):
        """
        Initialize the food predictor with the trained model
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
            print(f"✓ Model loaded successfully from {self.model_path}")
            
            # Load metadata
            metadata_path = 'models/model_metadata.pkl'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.class_names = metadata['class_names']
                self.image_size = metadata['image_size']
                print(f"✓ Model metadata loaded. Classes: {self.class_names}")
            else:
                # Fallback class names if metadata not available
                self.class_names = ['freshapples', 'freshbanana', 'freshoranges', 
                                  'rottenapples', 'rottenbanana', 'rottenoranges']
                print("⚠ Using default class names")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        """
        try:
            # Load and resize image
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize(self.image_size)
            
            # Convert to array and normalize
            img_array = np.array(image)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path):
        """
        Predict the class of an image
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Determine if fresh or rotten
            is_fresh = 'fresh' in predicted_class.lower()
            fruit_type = predicted_class.replace('fresh', '').replace('rotten', '').replace('apples', 'apple').replace('banana', 'banana').replace('oranges', 'orange')
            
            # Get all predictions for detailed output
            all_predictions = {}
            for i, class_name in enumerate(self.class_names):
                all_predictions[class_name] = float(predictions[0][i])
            
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_fresh': is_fresh,
                'fruit_type': fruit_type,
                'status': 'Fresh' if is_fresh else 'Rotten',
                'all_predictions': all_predictions
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return {'error': str(e)}
    
    def print_prediction(self, result):
        """
        Print prediction results in a nice format
        """
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return
        
        print(f"\n{'='*50}")
        print(f"🍎🍌🍊 FOOD DETECTION RESULT 🍎🍌🍊")
        print(f"{'='*50}")
        print(f"📁 Image: {result['image_path']}")
        print(f"🎯 Prediction: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"🍎 Fruit Type: {result['fruit_type'].title()}")
        print(f"✅ Status: {result['status']}")
        print(f"{'='*50}")
        
        print(f"\n📈 All Predictions:")
        for class_name, confidence in sorted(result['all_predictions'].items(), 
                                           key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {confidence:.2%}")


def main():
    """
    Main function for command line usage
    """
    parser = argparse.ArgumentParser(description='Food Detection Prediction')
    parser.add_argument('image_path', help='Path to the image file to predict')
    parser.add_argument('--model', default='models/food_detection_model.h5', 
                       help='Path to the trained model file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = FoodPredictor(args.model)
    
    # Make prediction
    result = predictor.predict(args.image_path)
    
    # Print results
    predictor.print_prediction(result)


if __name__ == "__main__":
    main()
