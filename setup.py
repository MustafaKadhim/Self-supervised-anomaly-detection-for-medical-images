from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="anomaly-detection-medical",
    version="1.0.0",
    author="Mustafa Kadhim",
    description="Self-supervised anomaly detection framework for medical images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MustafaKadhim/Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
