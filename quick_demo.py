#!/usr/bin/env python3
"""
Quick Demo of AI Media Detection System
Shows the speed and simplicity of the optimized detection
"""

import time
import tempfile
import numpy as np
from PIL import Image
from ai_detector import AIMediaDetector

def create_sample_image():
    """Create a sample image for testing"""
    # Create a random image
    image_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    image.save(temp_file.name, 'JPEG')
    return temp_file.name

def main():
    print("🚀 AI Media Detection - Speed Demo")
    print("=" * 50)
    
    # Initialize detector
    print("📡 Loading AI detector...")
    start_time = time.time()
    detector = AIMediaDetector()
    load_time = time.time() - start_time
    print(f"✅ Detector loaded in {load_time:.2f} seconds")
    
    # Create sample image
    print("\n📸 Creating sample image...")
    image_path = create_sample_image()
    print(f"✅ Sample image created: {image_path}")
    
    # Run detection
    print("\n🔍 Running AI detection...")
    start_time = time.time()
    result = detector.detect_ai_image(image_path)
    detection_time = time.time() - start_time
    
    print(f"✅ Detection completed in {detection_time:.2f} seconds")
    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    print("=" * 50)
    
    # Display clear results
    print(f"🎯 Verdict: {result['verdict']}")
    print(f"📈 Confidence: {result['confidence']:.1%}")
    print(f"⚡ Processing Speed: {detection_time:.2f}s")
    
    if result['analysis']:
        print(f"\n🔬 Analysis Factors:")
        for factor in result['analysis']:
            print(f"   • {factor}")
    
    # Clean up
    import os
    os.unlink(image_path)
    
    print("\n" + "=" * 50)
    print("✨ Demo completed successfully!")
    print("🚀 The system is now much faster and gives clear results!")
    print("💡 Try uploading your own images through the web interface.")

if __name__ == "__main__":
    main()