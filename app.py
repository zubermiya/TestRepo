import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from ai_detector import AIMediaDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize AI detector
detector = AIMediaDetector()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def is_video(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Detect if media is AI-generated
            if is_image(filename):
                result = detector.detect_ai_image(filepath)
            elif is_video(filename):
                result = detector.detect_ai_video(filepath)
            else:
                return jsonify({'error': 'Unsupported file type'}), 400
            
            return jsonify({
                'filename': filename,
                'file_type': 'image' if is_image(filename) else 'video',
                'ai_probability': result['ai_probability'],
                'confidence': result['confidence'],
                'analysis': result['analysis'],
                'metadata': result.get('metadata', {}),
                'success': True
            })
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def analyze_media():
    """Detailed analysis endpoint for advanced features"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Perform detailed analysis
        if is_image(filename):
            detailed_result = detector.detailed_image_analysis(filepath)
        elif is_video(filename):
            detailed_result = detector.detailed_video_analysis(filepath)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        return jsonify(detailed_result)
    
    except Exception as e:
        logger.error(f"Error in detailed analysis: {str(e)}")
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': detector.models_loaded})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)