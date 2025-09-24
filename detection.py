"""
Food Detection Model - Fresh vs Rotten Fruit Classification
This script trains a CNN model to classify fresh vs rotten fruits and saves it for app development.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FoodDetectionModel:
    def __init__(self, dataset_path='dataset', image_size=(224, 224), batch_size=32):
        """
        Initialize the Food Detection Model
        
        Args:
            dataset_path (str): Path to the dataset directory
            image_size (tuple): Target image size for the model
            batch_size (int): Batch size for training
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = []
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def get_class_names(self):
        """Get class names from the dataset structure"""
        train_path = os.path.join(self.dataset_path, 'train')
        if os.path.exists(train_path):
            self.class_names = sorted(os.listdir(train_path))
            print(f"Found classes: {self.class_names}")
        else:
            raise FileNotFoundError(f"Training directory not found at {train_path}")
        return self.class_names
    
    def create_data_generators(self):
        """
        Create data generators for training and validation with data augmentation
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2  # Use 20% of training data for validation
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        self.validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        # Test generator
        self.test_generator = test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'test'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        
        return self.train_generator, self.validation_generator, self.test_generator
    
    def create_model(self):
        """
        Create a CNN model using transfer learning with MobileNetV2
        """
        # Load pre-trained MobileNetV2 model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=(*self.image_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created successfully!")
        self.model.summary()
        return self.model
    
    def train_model(self, epochs=50):
        """
        Train the model with callbacks
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        print("Starting training...")
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate the model on test data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("Evaluating model on test data...")
        
        # Get predictions
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        
        # Get predictions for confusion matrix
        predictions = self.model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Generate classification report
        class_labels = list(self.test_generator.class_indices.keys())
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_labels
        )
        
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_classes, predicted_classes, class_labels)
        
        return test_accuracy, predictions
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """
        Save the model in multiple formats for different use cases
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("Saving model in multiple formats...")
        
        # 1. Save as H5 format (Keras format)
        self.model.save('models/food_detection_model.h5')
        print("✓ Saved as H5 format: models/food_detection_model.h5")
        
        # 2. Save as SavedModel format (TensorFlow format)
        self.model.save('models/food_detection_model_savedmodel')
        print("✓ Saved as SavedModel format: models/food_detection_model_savedmodel/")
        
        # 3. Save model weights only
        self.model.save_weights('models/food_detection_weights.h5')
        print("✓ Saved model weights: models/food_detection_weights.h5")
        
        # 4. Save model architecture as JSON
        model_json = self.model.to_json()
        with open('models/food_detection_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        print("✓ Saved model architecture: models/food_detection_architecture.json")
        
        # 5. Save class names and other metadata
        import pickle
        metadata = {
            'class_names': self.class_names,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'num_classes': len(self.class_names)
        }
        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print("✓ Saved model metadata: models/model_metadata.pkl")
        
        print("\nAll model formats saved successfully!")
    
    def predict_image(self, image_path):
        """
        Predict the class of a single image
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.image_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        return predicted_class, confidence
    
    def load_model(self, model_path='models/food_detection_model.h5'):
        """
        Load a pre-trained model
        """
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            
            # Load metadata
            metadata_path = 'models/model_metadata.pkl'
            if os.path.exists(metadata_path):
                import pickle
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.class_names = metadata['class_names']
                self.image_size = metadata['image_size']
                self.batch_size = metadata['batch_size']
                print(f"Model metadata loaded. Classes: {self.class_names}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")


def main():
    """
    Main function to train the food detection model
    """
    print("🍎🍌🍊 Food Detection Model Training 🍎🍌🍊")
    print("=" * 50)
    
    # Initialize the model
    food_detector = FoodDetectionModel()
    
    # Get class names
    food_detector.get_class_names()
    
    # Create data generators
    train_gen, val_gen, test_gen = food_detector.create_data_generators()
    
    # Create the model
    food_detector.create_model()
    
    # Train the model
    food_detector.train_model(epochs=30)
    
    # Evaluate the model
    test_accuracy, predictions = food_detector.evaluate_model()
    
    # Plot training history
    food_detector.plot_training_history()
    
    # Save the model
    food_detector.save_model()
    
    print(f"\n🎉 Training completed! Final test accuracy: {test_accuracy:.4f}")
    print("\nModel files saved in 'models/' directory:")
    print("  - food_detection_model.h5 (Keras format)")
    print("  - food_detection_model_savedmodel/ (TensorFlow format)")
    print("  - food_detection_weights.h5 (weights only)")
    print("  - food_detection_architecture.json (architecture)")
    print("  - model_metadata.pkl (metadata)")
    
    return food_detector


if __name__ == "__main__":
    # Train the model
    model = main()
    
    # Example: Predict on a single image (uncomment to use)
    # predicted_class, confidence = model.predict_image('path/to/your/image.jpg')
    # print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")
