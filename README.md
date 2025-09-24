# 🍎🍌🍊 Food Detection Model

A deep learning model to classify fresh vs rotten fruits using computer vision and transfer learning. The model can distinguish between fresh and rotten apples, bananas, and oranges.

## 📁 Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── train/
│   ├── freshapples/
│   ├── freshbanana/
│   ├── freshoranges/
│   ├── rottenapples/
│   ├── rottenbanana/
│   └── rottenoranges/
└── test/
    ├── freshapples/
    ├── freshbanana/
    ├── freshoranges/
    ├── rottenapples/
    ├── rottenbanana/
    └── rottenoranges/
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python detection.py
```

This will:
- Load and preprocess your dataset
- Create a CNN model using MobileNetV2 transfer learning
- Train the model with data augmentation
- Evaluate the model performance
- Save the model in multiple formats
- Generate training plots and confusion matrix

### 3. Use the Trained Model

#### Command Line Prediction
```bash
python predict.py path/to/your/image.jpg
```

#### Web Application
```bash
python app_example.py
```
Then open your browser to `http://localhost:5000`

## 📊 Model Performance

The model uses:
- **Architecture**: MobileNetV2 (transfer learning)
- **Input Size**: 224x224 pixels
- **Classes**: 6 classes (fresh/rotten for apples, bananas, oranges)
- **Data Augmentation**: Rotation, zoom, shift, flip
- **Validation**: 20% split from training data

## 📁 Output Files

After training, the following files will be created:

### Models (`models/` directory)
- `food_detection_model.h5` - Keras model format
- `food_detection_model_savedmodel/` - TensorFlow SavedModel format
- `food_detection_weights.h5` - Model weights only
- `food_detection_architecture.json` - Model architecture
- `model_metadata.pkl` - Model metadata and class names

### Visualizations (`plots/` directory)
- `training_history.png` - Training/validation accuracy and loss curves
- `confusion_matrix.png` - Confusion matrix for test data

## 🔧 Model Usage in Applications

### Python API
```python
from detection import FoodDetectionModel

# Load trained model
model = FoodDetectionModel()
model.load_model('models/food_detection_model.h5')

# Predict on image
predicted_class, confidence = model.predict_image('path/to/image.jpg')
print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")
```

### Web API
```python
# Start the Flask app
python app_example.py

# Make API calls
import requests
files = {'image': open('image.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/predict', files=files)
result = response.json()
```

## 📱 Mobile App Integration

The model can be integrated into mobile apps:

1. **TensorFlow Lite**: Convert the SavedModel to TFLite format
2. **ONNX**: Export to ONNX format for cross-platform deployment
3. **Core ML**: Convert for iOS apps
4. **TensorFlow.js**: For web applications

## 🎯 Model Classes

The model can classify:
- `freshapples` - Fresh apples
- `freshbanana` - Fresh bananas  
- `freshoranges` - Fresh oranges
- `rottenapples` - Rotten apples
- `rottenbanana` - Rotten bananas
- `rottenoranges` - Rotten oranges

## 🔄 Retraining

To retrain with new data:
1. Update your dataset structure
2. Run `python detection.py` again
3. The model will be retrained and saved

## 📈 Performance Tips

- **GPU**: Use GPU acceleration for faster training
- **Batch Size**: Adjust batch size based on your GPU memory
- **Epochs**: Monitor validation loss to prevent overfitting
- **Data Augmentation**: Already included for better generalization

## 🐛 Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size or image size
2. **Poor Performance**: Check dataset quality and balance
3. **Import Errors**: Install all requirements with `pip install -r requirements.txt`

### Model Loading Issues:
- Ensure all model files are in the `models/` directory
- Check that `model_metadata.pkl` exists for class names

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.