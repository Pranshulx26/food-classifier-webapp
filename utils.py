import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# Define class names
class_names = ['pizza', 'steak', 'sushi']
out_features = len(class_names)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def load_model(model_path: str) -> torch.nn.Module:
    """
    Loads a saved ConvNeXt-Tiny model from the given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found at {model_path}')

    model = create_convnext_tiny(out_features, device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])  
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def predict_image(image_path: str, model: torch.nn.Module, confidence_threshold: float = 0.5):
    """
    Predicts the class of a single image using the provided model.
    """
    try:
        image = Image.open(image_path).convert('RGB')

        # Use ConvNeXt-Tiny weights transform
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        image_transform = weights.transforms()
        image_tensor = image_transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        confidence_score = confidence.item()
        predicted_class = class_names[pred_idx.item()]

        if confidence_score >= confidence_threshold:
            return predicted_class, confidence_score
        else:
            return "Image not classified", confidence_score

    except Exception as e:
        print(f'Error during prediction: {e}')
        return "Error processing image", 0.0
