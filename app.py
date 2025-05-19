import os
import uuid
import torch
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils import load_model, predict_image

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set maximum content length for file uploads (5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Load the ConvNeXt-Tiny model at startup
MODEL_PATH = os.path.join('model', 'convnext_tiny_10_epochs_20250519_193505.pt')
print("Absolute MODEL_PATH:", os.path.abspath(MODEL_PATH))
print("Exists:", os.path.isfile(MODEL_PATH))

model = None
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    print(f"Warning: Model file not found at {MODEL_PATH}")
    print("The app will start, but prediction functionality will not work")
except Exception as e:
    print(f"Error loading model: {e}")

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400

    try:
        # Save file with unique name
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Ensure model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check your model path and loading logic.'}), 500

        # Predict the class
        predicted_class, confidence_score = predict_image(filepath, model)

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence_score * 100, 2),
            'image_path': filepath
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 5MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True)
