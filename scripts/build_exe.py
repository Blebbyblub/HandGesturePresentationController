"""
Build Script for Hand Gesture Presentation Controller
Creates a standalone .exe file using PyInstaller

Run this script with: python build_exe.py
"""

import subprocess
import sys
import os
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not present"""
    try:
        import PyInstaller
        print("✓ PyInstaller is already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed successfully")

def build_executable():
    """Build the executable using PyInstaller"""
    
    # Get the project root directory (parent of scripts/)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    controller_path = project_root / "controller.py"
    artifacts_dir = project_root / "artifacts"
    
    # Check if controller.py exists
    if not controller_path.exists():
        print(f"❌ Error: {controller_path} not found!")
        return False
    
    # Check if model exists
    model_path = artifacts_dir / "gesture_svm_v3.pkl"
    if not model_path.exists():
        # Try backup model
        model_path = artifacts_dir / "gesture_svm.pkl"
        if not model_path.exists():
            print("❌ Error: No model file found in artifacts/")
            return False
    
    print(f"✓ Found controller.py at: {controller_path}")
    print(f"✓ Found model at: {model_path}")
    
    # PyInstaller command
    # --onefile: Create a single .exe file
    # --noconsole: Hide console window (shows only the OpenCV window)
    # --add-data: Include the artifacts folder
    # --name: Name of the output executable
    
    # Determine the correct separator for --add-data (';' on Windows, ':' on Unix)
    separator = ';' if sys.platform == 'win32' else ':'
    
    # Find MediaPipe package location for its model files
    import mediapipe
    mediapipe_path = Path(mediapipe.__file__).parent
    
    pyinstaller_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--noconsole",
        f"--add-data={artifacts_dir}{separator}artifacts",
        f"--add-data={mediapipe_path}/modules{separator}mediapipe/modules",
        "--name=HandGestureController",
        "--icon=NONE",  # You can add an icon file here if you have one
        "--clean",
        str(controller_path)
    ]
    
    print("\n" + "="*60)
    print("Building executable...")
    print("This may take a few minutes...")
    print("="*60 + "\n")
    
    try:
        subprocess.check_call(pyinstaller_cmd, cwd=str(script_dir))
        
        exe_path = script_dir / "dist" / "HandGestureController.exe"
        if exe_path.exists():
            print("\n" + "="*60)
            print("✓ BUILD SUCCESSFUL!")
            print("="*60)
            print(f"\nExecutable created at:")
            print(f"  {exe_path}")
            print("\nTo run the application:")
            print("  1. Double-click HandGestureController.exe")
            print("  2. Or run from command line: .\\dist\\HandGestureController.exe")
            print("\nNote: Make sure your webcam is connected!")
            return True
        else:
            print("❌ Build completed but .exe not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with error: {e}")
        return False

def main():
    print("="*60)
    print("HAND GESTURE CONTROLLER - BUILD TOOL")
    print("="*60)
    print()
    
    # Step 1: Install PyInstaller
    install_pyinstaller()
    
    # Step 2: Build executable
    success = build_executable()
    
    if success:
        print("\n✓ You can now use HandGestureController.exe for your demo!")
    else:
        print("\n❌ Build failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
