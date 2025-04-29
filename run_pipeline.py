import os
import subprocess
import sys
from pathlib import Path

def check_environment():
    """Check if all required packages are installed"""
    required_packages = [
        'tensorflow',
        'nibabel',
        'scipy',
        'matplotlib',
        'opencv-python',
        'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nPlease install missing packages using:")
        print("pip install " + " ".join(missing_packages))
        sys.exit(1)

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "Data/raw/MR",
        "Data/raw/CT",
        "Data/Preprocessed",
        "checkpoints",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running {description}...")
    print(f"{'='*50}")
    
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"✓ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}:")
        print(e)
        sys.exit(1)

def main():
    # Check environment
    print("Checking environment...")
    check_environment()
    
    # Create directory structure
    print("\nCreating directory structure...")
    create_directory_structure()
    
    # Run pipeline steps
    steps = [
        ("Preprocessing 2/Split Slice to Preprocess", "Slice splitting"),
        ("Preprocessing 2/Preprocess_OK", "Data preprocessing"),
        ("train.py", "Model training"),
        ("test.py", "Model evaluation")
    ]
    
    for script, description in steps:
        run_script(script, description)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 