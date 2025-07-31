#!/usr/bin/env python3
"""
Setup script for AI Media Detection System
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'static/css', 'static/js', 'templates']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            print(f"⚠️  Warning: Low memory detected ({memory_gb:.1f}GB). 4GB+ recommended")
        else:
            print(f"✅ Memory: {memory_gb:.1f}GB")
    except ImportError:
        print("ℹ️  Install psutil to check memory requirements: pip install psutil")
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CUDA not available - using CPU (slower processing)")
    except ImportError:
        print("ℹ️  PyTorch not yet installed")

def setup_environment():
    """Set up the development environment"""
    print("🛠️  Setting up AI Media Detection System...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Check system requirements
    check_system_requirements()
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: python app.py")
    print("2. Open your browser to: http://localhost:5000")
    print("3. Upload an image or video to test the system")
    print("\n📚 For more information, see README.md")

def run_tests():
    """Run basic system tests"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import torch
        import cv2
        import PIL
        import flask
        print("✅ All core libraries imported successfully")
        
        # Test model loading (basic check)
        from ai_detector import AIMediaDetector
        print("✅ AI detector module loads successfully")
        
        print("✅ All tests passed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: python setup.py install")
    except Exception as e:
        print(f"❌ Test error: {e}")

def clean_uploads():
    """Clean uploaded files"""
    uploads_dir = 'uploads'
    if os.path.exists(uploads_dir):
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            try:
                os.remove(file_path)
                print(f"🗑️  Removed: {filename}")
            except Exception as e:
                print(f"❌ Error removing {filename}: {e}")
        print("✅ Upload directory cleaned")
    else:
        print("📁 Upload directory doesn't exist")

def show_help():
    """Show help information"""
    print("🔧 AI Media Detection Setup Tool")
    print("=" * 40)
    print("Usage: python setup.py [command]")
    print("\nCommands:")
    print("  install    - Install dependencies and set up environment")
    print("  test       - Run basic system tests")
    print("  clean      - Clean uploaded files")
    print("  help       - Show this help message")
    print("\nExamples:")
    print("  python setup.py install")
    print("  python setup.py test")
    print("  python setup.py clean")

def main():
    """Main setup function"""
    if len(sys.argv) < 2:
        setup_environment()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'install':
        setup_environment()
    elif command == 'test':
        run_tests()
    elif command == 'clean':
        clean_uploads()
    elif command == 'help':
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()