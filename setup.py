import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="focal_frequency_loss",
    version="0.1.0",
    author="Liming Jiang",
    author_email="liming002@ntu.edu.sg",
    description="Focal Frequency Loss for Image Reconstruction and Synthesis - Official PyTorch Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EndlessSora/focal-frequency-loss",
    packages=['focal_frequency_loss'],
    python_requires=">=3.5",
    install_requires=["torch<=1.7.1,>=1.1.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
