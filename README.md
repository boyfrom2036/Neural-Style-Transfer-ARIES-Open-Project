# Neural-Style-Transfer-ARIES-Open-Project
Project Overview
This project implements Neural Style Transfer using the VGG19 network in PyTorch. The goal of neural style transfer is to combine the content of one image with the style of another, resulting in a new image that retains the content of the first image but appears to be "painted" in the style of the second image. This project demonstrates how to achieve this using a pre-trained VGG19 model, leveraging its deep convolutional layers to extract content and style representations.

Prerequisites
Before running the project, ensure you have the following installed:

Python 3.6+
PyTorch
torchvision
Pillow
NumPy
Matplotlib

Parameters
--content: Path to the content image.
--style: Path to the style image.
--output: Path where the output image will be saved.
--iterations: Number of iterations for optimization.
--content_weight: Weight for the content loss.
--style_weight: Weight for the style loss.
--lr: Learning rate for the optimizer.

Dependencies
The project requires the following libraries and dependencies:

Python 3.6+: The programming language used for this project.
PyTorch: A deep learning framework used to build and train the model.
torchvision: Provides easy access to pre-trained models and image transformations.
Pillow: A Python Imaging Library (PIL) fork used for image processing.
NumPy: A library for numerical computations in Python.
Matplotlib: A plotting library used for visualizing images.

To install these dependencies, run:

pip install torch torchvision pillow numpy matplotlib
