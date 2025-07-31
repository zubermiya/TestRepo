#!/usr/bin/env python3
"""
AI Media Detection System Runner
Simple script to start the application with proper setup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_setup():
    """Check if the system is properly set up"""
    required_files = ['app.py', 'ai_detector.py', 'requirements.txt']
    required_dirs = ['templates', 'static']
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_files or missing_dirs:
        print("❌ Setup incomplete!")
        if missing_files:
            print(f"Missing files: {', '.join(missing_files)}")
        if missing_dirs:
            print(f"Missing directories: {', '.join(missing_dirs)}")
        print("\nRun: python setup.py install")
        return False
    
    return True

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import torch
        import cv2
        import PIL
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: python setup.py install")
        return False

def start_application():
    """Start the Flask application"""
    print("🚀 Starting AI Media Detection System...")
    print("=" * 50)
    print("📍 Server will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Check the logs above for more details")

def main():
    """Main runner function"""
    print("🔍 AI Media Detection System")
    print("College Project - Media Authenticity Verification")
    print()
    
    # Check if setup is complete
    if not check_setup():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()