import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Load the VGG19 model pre-trained on ImageNet data
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# Function to load and preprocess the image
def load_image(img, max_size=400, shape=None):
    image = Image.open(img).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image.to(device)

# Function to extract features from specified layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # Content layer
                  '28': 'conv5_1'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

# Function to compute the Gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Modified content loss function
def modified_compute_content_loss(target, content):
    difference = target - content
    difference[difference < 0] = 0
    loss = torch.sum(difference ** 2) / 2
    return loss

# Style loss function
def compute_style_loss(target, style_gram):
    _, d, h, w = target.size()
    target_gram = gram_matrix(target)
    loss = torch.mean((target_gram - style_gram) ** 2)
    return loss

# Function to convert the tensor back to an image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

# Streamlit UI
st.title("Neural Style Transfer")

content_file = st.file_uploader("Choose a content image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Choose a style image", type=["jpg", "jpeg", "png"])

if content_file is not None and style_file is not None:
    content = load_image(content_file)
    style = load_image(style_file, shape=content.shape[-2:])

    content_layers = {'21': 'conv4_2'}
    style_layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}

    content_features = get_features(content, vgg, content_layers)
    style_features = get_features(style, vgg, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)

    content_weight = 1
    style_weight = 1e6

    layer_weights = {
        'conv1_1': 1/5,
        'conv2_1': 1/5,
        'conv3_1': 1/5,
        'conv4_1': 1/5,
        'conv5_1': 1/5
    }

    if st.button("Start Style Transfer"):
        for i in range(1, 3001):
            target_features = get_features(target, vgg, {**content_layers, **style_layers})
            content_loss = modified_compute_content_loss(target_features['conv4_2'], content_features['conv4_2'])

            style_loss = 0
            for layer in style_layers.values():
                target_feature = target_features[layer]
                style_gram = style_grams[layer]
                style_loss += layer_weights[layer] * compute_style_loss(target_feature, style_gram)

            total_loss = content_weight * content_loss + style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                st.write(f'Iteration {i}: Total loss: {total_loss.item():.4f}')

        final_image = im_convert(target)
        st.image(final_image, caption="Stylized Image", use_column_width=True)
