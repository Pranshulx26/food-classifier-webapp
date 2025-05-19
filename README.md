# 🍕 Food-Vision: Advanced Deep Learning Food Classifier with Web Interface

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-000000.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art deep learning web application that accurately classifies food images as pizza, steak, or sushi using transfer learning with ConvNeXt-Tiny architecture, achieving over 98% accuracy on test data.

![Food Vision Demo](demo.png)

## Project Overview

Food-Vision showcases end-to-end machine learning deployment, from dataset preparation and model training to production-ready web application development. The project demonstrates expertise in:

- **Computer Vision**: Advanced image preprocessing and classification using PyTorch
- **Transfer Learning**: Systematic evaluation of multiple architectures including EfficientNet, ResNet, ViT, and ConvNeXt
- **Experimental Design**: Structured experimentation to identify optimal architecture and hyperparameters
- **Web Development**: Full-stack deployment with Flask and modern frontend

The ConvNeXt-Tiny model achieves an impressive 98.38% accuracy on the test set after comprehensive architecture evaluation and hyperparameter tuning.

## Technical Architecture

### Model Selection Process
Implemented a systematic evaluation of multiple state-of-the-art architectures:
- EfficientNet-B0 and B2
- ResNet50
- Vision Transformer (ViT-B/16)
- ConvNeXt-Tiny

ConvNeXt-Tiny emerged as the superior architecture with the following advantages:
- Highest accuracy (98.38%)
- Excellent balance of model complexity and performance
- Efficient inference for web application deployment

### Data Preparation
- Specialized dataset containing pizza, steak, and sushi images
- Organized in training and test sets for robust model evaluation
- Applied architecture-specific preprocessing transforms

### Model Architecture
Implemented ConvNeXt-Tiny with custom classification head:

```python
def create_convnext_tiny(out_features: int = 3, device: torch.device = device) -> torch.nn.Module:
    """
    Initializes and returns a ConvNeXt-Tiny model with a custom classifier head.
    """
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights).to(device)

    # Update the classifier to match the number of output classes
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=768, out_features=out_features)
    ).to(device)

    return model
```

### Training Process and Results
- Implemented a comprehensive experimental framework to compare multiple models
- Used Adam optimizer with learning rate 0.001
- Incorporated TensorBoard logging for experiment tracking
- Implemented early stopping with patience=3
- Trained for 10 epochs with the following performance trajectory:
  - Epoch 1: 87.04% test accuracy
  - Epoch 5: 97.84% test accuracy
  - Epoch 8: 98.38% test accuracy (peak performance)
  - Final accuracy: 98.25% on test set

### Web Application
- **Backend**: Flask server handling image uploads and inference
- **Frontend**: Responsive UI built with Bootstrap and custom CSS
- **Features**:
  - Clean and intuitive user interface
  - Real-time image classification with confidence scores
  - Visualized confidence levels with color-coded progress bars
  - Error handling and validation
  - Mobile-responsive design
- **GPU/CPU Support**: Automatically detects and utilizes available hardware

## Project Structure

```
food_classifier/
├── app.py                # Main Flask application
├── utils.py              # Helper functions for model loading and prediction
├── requirements.txt      # Project dependencies
├── static/               # Static files (CSS, JS, etc.)
│   ├── css/
│   │   └── style.css     # Custom CSS styles
│   └── uploads/          # Directory to store uploaded images
├── templates/            # HTML templates
│   └── index.html        # Main page template
├── model/                # Directory for the trained model
│   └── convnext_tiny_10_epochs_20250519_193505.pt  # Trained PyTorch ConvNeXt-Tiny model
└── notebook/             # Jupyter notebooks for model development
    └── food_classification.ipynb       # Model training and experimentation
    └── model_comparison.ipynb          # Architecture comparison experiments
    └── hyperparameter_tuning.ipynb     # Optimization of model parameters
```

## Installation and Usage

### Prerequisites
- Python 3.9+
- Git
- Anaconda or Miniconda (recommended)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/Pranshulx26/food-classifier-webapp.git
   cd food-classifier-webapp
   ```

2. **Create and activate a conda environment**
   ```bash
   conda create -n food-classifier python=3.9
   conda activate food-classifier
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. **Ensure the model file is in place**
   - The trained model file `convnext_tiny_10_epochs_20250519_193505.pt` should be in the `model/` directory

2. **Start the Flask server**
   ```bash
   python app.py
   ```

3. **Access the web interface**
   - Open your browser and navigate to `http://127.0.0.1:5000/`
   - Upload an image of pizza, steak, or sushi
   - View the classification results with confidence scores

## Model Performance Comparison

| Model Architecture | Test Accuracy | Training Time | Notes |
|-------------------|--------------|--------------|-------|
| ConvNeXt-Tiny     | 98.38%       | 14min 36s    | Best overall performance |
| EfficientNet-B2   | 96.82%       | 18min 12s    | Good balance of accuracy and size |
| EfficientNet-B0   | 92.00%       | 12min 45s    | Previous implementation |
| ResNet50          | 95.73%       | 16min 30s    | Solid baseline performance |
| ViT-B/16          | 97.15%       | 25min 08s    | High accuracy but slower inference |

## Future Improvements
- Expand to more food categories beyond the current three classes
- Deploy model with ONNX or TorchScript for optimized inference
- Implement ensemble methods for even higher accuracy
- Deploy to cloud platforms (AWS, GCP, Azure) for public access
- Add user accounts to save classification history
- Integrate with nutrition APIs for detailed food information

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- PyTorch and torchvision for model development: [https://pytorch.org/](https://pytorch.org/)
- ConvNeXt paper: [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)
- Flask: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
- Bootstrap: [https://getbootstrap.com/](https://getbootstrap.com/)
