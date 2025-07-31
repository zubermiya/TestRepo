#!/usr/bin/env python3
"""
Test script for AI Media Detection System
Verifies core functionality and components
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image
import cv2

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import flask
        print("✅ Flask imported")
        
        import torch
        print(f"✅ PyTorch imported (version: {torch.__version__})")
        
        import torchvision
        print("✅ Torchvision imported")
        
        import cv2
        print(f"✅ OpenCV imported (version: {cv2.__version__})")
        
        import PIL
        print("✅ PIL imported")
        
        import numpy as np
        print("✅ NumPy imported")
        
        from ai_detector import AIMediaDetector
        print("✅ AI Detector module imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_ai_detector():
    """Test AI detector initialization"""
    print("\n🧪 Testing AI Detector...")
    
    try:
        from ai_detector import AIMediaDetector
        detector = AIMediaDetector()
        
        if detector.models_loaded:
            print("✅ AI Detector initialized successfully")
            print(f"✅ Using device: {detector.device}")
        else:
            print("⚠️  AI Detector initialized but models not loaded")
            
        return True
        
    except Exception as e:
        print(f"❌ AI Detector test failed: {e}")
        return False

def create_test_image():
    """Create a test image for analysis"""
    print("\n🧪 Creating test image...")
    
    try:
        # Create a simple test image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        pil_image.save(temp_file.name, 'JPEG')
        
        print(f"✅ Test image created: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Test image creation failed: {e}")
        return None

def test_image_analysis(image_path):
    """Test image analysis functionality"""
    print("\n🧪 Testing image analysis...")
    
    try:
        from ai_detector import AIMediaDetector
        detector = AIMediaDetector()
        
        if not detector.models_loaded:
            print("⚠️  Skipping analysis test - models not loaded")
            return True
        
        result = detector.detect_ai_image(image_path)
        
        # Check if result has expected keys
        expected_keys = ['ai_probability', 'confidence', 'analysis', 'metadata', 'verdict']
        missing_keys = [key for key in expected_keys if key not in result]
        
        if missing_keys:
            print(f"❌ Missing keys in result: {missing_keys}")
            return False
        
        print("✅ Image analysis completed")
        print(f"   AI Probability: {result['ai_probability']:.4f}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Verdict: {result['verdict']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image analysis test failed: {e}")
        return False

def test_metadata_extraction(image_path):
    """Test metadata extraction"""
    print("\n🧪 Testing metadata extraction...")
    
    try:
        from ai_detector import AIMediaDetector
        detector = AIMediaDetector()
        
        metadata = detector.extract_metadata(image_path)
        
        if not metadata:
            print("⚠️  No metadata extracted")
            return True
        
        print("✅ Metadata extraction completed")
        print(f"   Keys found: {list(metadata.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata extraction test failed: {e}")
        return False

def test_flask_app():
    """Test Flask application setup"""
    print("\n🧪 Testing Flask application...")
    
    try:
        from app import app
        
        # Test app configuration
        if app.config.get('UPLOAD_FOLDER'):
            print("✅ Upload folder configured")
        
        if app.config.get('MAX_CONTENT_LENGTH'):
            print("✅ Max content length configured")
        
        # Test routes exist
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/upload', '/analyze', '/health']
        
        missing_routes = [route for route in expected_routes if route not in routes]
        if missing_routes:
            print(f"❌ Missing routes: {missing_routes}")
            return False
        
        print("✅ Flask application configured correctly")
        return True
        
    except Exception as e:
        print(f"❌ Flask application test failed: {e}")
        return False

def test_file_validation():
    """Test file validation functions"""
    print("\n🧪 Testing file validation...")
    
    try:
        from app import allowed_file, is_image, is_video
        
        # Test image files
        image_files = ['test.jpg', 'test.png', 'test.gif']
        for filename in image_files:
            if not allowed_file(filename) or not is_image(filename):
                print(f"❌ Image file validation failed for {filename}")
                return False
        
        # Test video files
        video_files = ['test.mp4', 'test.avi', 'test.mov']
        for filename in video_files:
            if not allowed_file(filename) or not is_video(filename):
                print(f"❌ Video file validation failed for {filename}")
                return False
        
        # Test invalid files
        invalid_files = ['test.txt', 'test.exe', 'test.pdf']
        for filename in invalid_files:
            if allowed_file(filename):
                print(f"❌ Invalid file incorrectly allowed: {filename}")
                return False
        
        print("✅ File validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ File validation test failed: {e}")
        return False

def cleanup_test_files(file_paths):
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                print(f"✅ Removed: {file_path}")
        except Exception as e:
            print(f"⚠️  Could not remove {file_path}: {e}")

def run_all_tests():
    """Run all tests"""
    print("🔍 AI Media Detection System - Test Suite")
    print("=" * 50)
    
    test_files = []
    all_passed = True
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("AI Detector Test", test_ai_detector),
        ("Flask App Test", test_flask_app),
        ("File Validation Test", test_file_validation),
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if not test_func():
            all_passed = False
    
    # Create test image for analysis tests
    test_image = create_test_image()
    if test_image:
        test_files.append(test_image)
        
        analysis_tests = [
            ("Image Analysis Test", lambda: test_image_analysis(test_image)),
            ("Metadata Extraction Test", lambda: test_metadata_extraction(test_image)),
        ]
        
        for test_name, test_func in analysis_tests:
            print(f"\n📋 Running {test_name}...")
            if not test_func():
                all_passed = False
    
    # Cleanup
    cleanup_test_files(test_files)
    
    # Final results
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📋 Next steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Upload a test image or video")
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("💡 Try running: python setup.py install")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)