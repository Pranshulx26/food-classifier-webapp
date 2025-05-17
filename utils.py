import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# Define class names 
class_names = ['pizza', 'steak', 'sushi']

def load_model(model_path):
    """
    Loads a pre-trained EfficientNet-B0 model and its saved state dictionary.

    Args:
        model_path (str): Path to the saved model file (.pth).

    Returns:
        torch.nn.Module: The loaded EfficientNet-B0 model.

    Raises:
        FileNotFoundError: If the model file is not found.
        Exception: If there's an error loading the model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found at {model_path}')

    # Initialize EfficientNet-B0 with pretrained weights
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Update the classifier head to match the number of classes in your problem
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),  # Keep dropout if you used it
        nn.Linear(in_features=1280, out_features=len(class_names))
    )

    # Load the model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))

        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

    # Set the model to evaluation mode
    model.eval()
    return model


def predict_image(image_path, model, confidence_threshold=0.5):
    """
    Predicts the class of an image using the provided pre-trained EfficientNet-B0 model.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): The pre-trained EfficientNet-B0 model.
        confidence_threshold (float, optional): Minimum confidence score for a valid prediction. Defaults to 0.7.

    Returns:
        tuple: (predicted_class, confidence_score) if confidence is above the threshold,
               otherwise ("Image not classified", confidence_score).
               Returns ("Error processing image", 0.0) on error.
    """
    try:
        # Open and process the image
        image = Image.open(image_path).convert('RGB')

        # Get the transforms from the model weights
        weights = models.EfficientNet_B0_Weights.DEFAULT
        auto_transforms = weights.transforms()
        image_tensor = auto_transforms(image).unsqueeze(0)


        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        # Convert to Python scalars
        confidence_score = confidence.item()
        predicted_idx = prediction.item()

        # Check if confidence is above threshold
        if confidence_score >= confidence_threshold:
            predicted_class = class_names[predicted_idx]
            return predicted_class, confidence_score
        else:
            return 'Image not classified', confidence_score

    except Exception as e:
        print(f'Error during prediction: {e}')
        return "Error processing image", 0.0




