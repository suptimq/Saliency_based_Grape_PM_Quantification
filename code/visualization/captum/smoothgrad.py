#!/usr/bin/env python
# coding: utf-8

# # Model Interpretation for Pretrained ResNet Model

# This notebook demonstrates how to apply model interpretability algorithms on pretrained ResNet model using a handpicked image and visualizes the attributions for each pixel by overlaying them on the image.
# 
# The interpretation algorithms that we use in this notebook are `Integrated Gradients` (w/ and w/o noise tunnel),  `GradientShap`, and `Occlusion`. A noise tunnel allows to smoothen the attributions after adding gaussian noise to each input sample.
#   
#   **Note:** Before running this tutorial, please install the torchvision, PIL, and matplotlib packages.

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


# ## 1- Loading the model and the dataset
# 

# Loads pretrained Resnet model and sets it to eval mode

# In[2]:
device = 'cuda:0'

def load_model():
    """
        Load pretrained VGG16
    """
    model_dir = Path('/media/cornell/Data/tq42/Hyphal_2020/results/models')
    timestamp = 'VGG_Dec03_23-27-35'

    model = models.vgg16(pretrained=True, num_classes=1000)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                     2, bias=True)
    last_conv_layer = model.features[28]
    # Load checkpoint
    model_path = model_dir / timestamp / 'VGG_model_ep167'
    # print(model_path)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, last_conv_layer

# model = models.resnet18(pretrained=True)
model, _ = load_model()
model = model.eval()


# Downloads the list of classes/labels for ImageNet dataset and reads them into the memory

# In[3]:


# get_ipython().system('wget -P $HOME/.torch/models https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')


# In[4]:


# labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
# with open(labels_path) as json_data:
#     idx_to_labels = json.load(json_data)

idx_to_labels = {'0': 'Healthy', '1': 'Infected'}
# Defines transformers and normalizing functions for the image.
# It also loads an image from the `img/resnet/` folder that will be used for interpretation purposes.

# In[5]:


transform = transforms.Compose([
#  transforms.Resize(256),
#  transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize((0.5, ), (0.5, ))
])

transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

# img = Image.open('swan-3299528_1280.jpg')
img = Image.open('../test_images/tray1_139-4510009_Vertical_patch_14.png')

transformed_img = transform(img)

# input = transform_normalize(transformed_img)
# input = input.unsqueeze(0).to(device)
input = transformed_img.unsqueeze(0)


# Predict the class of the input image

# In[6]:


output = model(input)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

pred_label_idx.squeeze_()
# predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
predicted_label = idx_to_labels[str(pred_label_idx.item())]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')


# ## 2- Gradient-based attribution

# Let's compute attributions using Integrated Gradients and visualize them on the image. Integrated gradients computes the integral of the gradients of the output of the model for the predicted class `pred_label_idx` with respect to the input image pixels along the path from the black image to our input image.

# In[7]:


print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

integrated_gradients = IntegratedGradients(model)
attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)




# Let's visualize the image and corresponding attributions by overlaying the latter on the image.

# In[8]:


default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)

print('finish')
# Let us compute attributions using Integrated Gradients and smoothens them across multiple images generated by a <em>noise tunnel</em>. The latter adds gaussian noise with a std equals to one, 10 times (n_samples=10) to the input. Ultimately, noise tunnel smoothens the attributions across `n_samples` noisy samples using `smoothgrad_sq` technique. `smoothgrad_sq` represents the mean of the squared attributions across `n_samples` samples.

# In[9]:


noise_tunnel = NoiseTunnel(integrated_gradients)

attributions_ig_nt = noise_tunnel.attribute(input, n_samples=50, stdevs=0.15, nt_type='smoothgrad', target=pred_label_idx)
a = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True)


# Finally, let us use `GradientShap`, a linear explanation model which uses a distribution of reference samples (in this case two images) to explain predictions of the model. It computes the expectation of gradients for an input which was chosen randomly between the input and a baseline. The baseline is also chosen randomly from given baseline distribution.

# In[10]:


torch.manual_seed(0)
np.random.seed(0)

gradient_shap = GradientShap(model)

# Defining baseline distribution of images
rand_img_dist = torch.cat([input * 0, input * 1])

attributions_gs = gradient_shap.attribute(input,
                                          n_samples=50,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=pred_label_idx)
_ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      cmap=default_cmap,
                                      show_colorbar=True)


# ## 3- Occlusion-based attribution

# Now let us try a different approach to attribution. We can estimate which areas of the image are critical for the classifier's decision by occluding them and quantifying how the decision changes.
# 
# We run a sliding window of size 15x15 (defined via `sliding_window_shapes`) with a stride of 8 along both image dimensions (a defined via `strides`). At each location, we occlude the image with a baseline value of 0 which correspondes to a gray patch (defined via `baselines`).
# 
# **Note:** this computation might take more than one minute to complete, as the model is evaluated at every position of the sliding window.

# In[11]:


occlusion = Occlusion(model)

attributions_occ = occlusion.attribute(input,
                                       strides = (3, 8, 8),
                                       target=pred_label_idx,
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)


# Let us visualize the attribution, focusing on the areas with positive attribution (those that are critical for the classifier's decision):

# In[12]:


_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )


# The upper part of the goose, especially the beak, seems to be the most critical for the model to predict this class.
# 
# We can verify this further by occluding the image using a larger sliding window:

# In[13]:


occlusion = Occlusion(model)

attributions_occ = occlusion.attribute(input,
                                       strides = (3, 50, 50),
                                       target=pred_label_idx,
                                       sliding_window_shapes=(3,60, 60),
                                       baselines=0)

_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )

