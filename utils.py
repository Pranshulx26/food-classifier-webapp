import os 
import torch 
import torchvision.transforms as transforms 
from PIL import Image 
from torch import nn 
import torch.nn.functional as F 

# Define the model architecture 
class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)  # Dropout after first pool 0.5 -> 0.3
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)  # Dropout after second pool 0.5 -> 0.3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 16 * 16, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)


class_names = ['pizza', 'steak', 'sushi']

# Define the image transformation 
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# Function to load the trained model
def load_model(model_path):
    """
    Load the trained PyTorch model
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: Loaded PyTorch model
    """
    # Check if model file exist 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found at {model_path}')
    
    # Initialize the model architecture
    model = TinyVGG(input_shape=3, hidden_units=64, output_shape=len(class_names))

    # Load the model weights 
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")
    
    # Set the model to evaluation mode 
    model.eval()

    return model 


# Functioni to predict the class of an image 
def predict_image(image_path, model, confidence_threshold=0.5):
    """
    Predict the class of an image using the trained model
    
    Args:
        image_path (str): Path to the image file
        model: PyTorch model
        confidence_threshold (float): Threshold for prediction confidence
        
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    try:
        # Open and process the image 
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0) 

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
        print(f'Error during predictions: {e}')
        return "Error processing image", 0.0
    


