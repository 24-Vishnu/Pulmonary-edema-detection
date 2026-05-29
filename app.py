import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Load the model
MODEL_PATH = 'models/pulmonary_edema_model_final.h5'
try:
    # Fix for warnings - suppress TensorFlow warnings temporarily
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Define custom metrics with exact names from warning message
    custom_objects = {
        'auc_1': tf.keras.metrics.AUC(name='auc_1'),
        'recall_1': tf.keras.metrics.Recall(name='recall_1'),
        'precision_1': tf.keras.metrics.Precision(name='precision_1')
    }
    
    # Load model with these custom metrics
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    
    # Recompile with only basic metrics to avoid warnings
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    logger.info("Model loaded successfully")
    
    class_names = ['no edema', 'edema']
    logger.info(f"Using class names: {class_names}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    class_names = ['no edema', 'edema']

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocess image function
def preprocess_image(image):
    # Convert to RGB (in case of grayscale X-rays)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to the input size expected by the model
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Save the uploaded image
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        img.save(file_path)
        
        # Preprocess the image
        processed_img = preprocess_image(img)
        
        # Make prediction
        prediction = model.predict(processed_img, verbose=0)  # Suppress prediction verbose output
        probability = float(prediction[0][0])
        
        # FIX: Invert the prediction logic to correct the classification
        # If probability > 0.5 means "no edema" in the original training,
        # but we want probability > 0.5 to mean "edema" in the UI
        predicted_class = class_names[0] if probability > 0.5 else class_names[1]
        result = "Normal (No Pulmonary Edema)" if probability > 0.5 else "Pulmonary Edema Detected"
        
        # Also invert the probability for correct confidence display
        display_probability = probability if probability > 0.5 else 1 - probability
        
        logger.info(f"Raw probability: {probability}, Inverted prediction: {predicted_class}")
        
        # Return the prediction
        return jsonify({
            'result': result,
            'predicted_class': predicted_class,
            'probability': display_probability,
            'filename': file.filename,
            'file_path': f"/static/uploads/{file.filename}"
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
