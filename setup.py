from setuptools import setup, find_packages

setup(
    name="mri_to_ct_dcnn",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow==1.15.0",
        "numpy",
        "opencv-python",
        "SimpleITK",
        "scipy",
        "matplotlib",
        "scikit-image",
        "pillow"
    ],
    author="Cheng-Bin Jin",
    author_email="sbkim0407@gmail.com",
    description="MRI to CT conversion using DCNN TensorFlow implementation",
) 