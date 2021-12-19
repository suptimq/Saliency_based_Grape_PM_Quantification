import os

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from torchvision import models
from torchvision import transforms

from captum.attr import visualization as viz
from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution

# Default device
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(model_type):
    """
        Load pretrained SqueezeNet
    """
    model_dir = '/Users/tim/BB_analysis/models'
    # Load pretrained model
    if model_type == 'SqueezeNet':
        model = models.squeezenet1_1(pretrained=False, num_classes=1000)
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels,
                                        2,
                                        kernel_size=(1, 1),
                                        stride=(1, 1))
        last_conv_layer = model.features[12].expand3x3
        # Load checkpoint
        model_path = os.path.join(model_dir, 'SqueezeNet_model_ep190')
    elif model_type == 'GoogleNet':
        model = models.googlenet(pretrained=False, num_classes=1000)
        model.fc = nn.Linear(model.fc.in_features, 2, bias=True)
        # Load checkpoint
        model_path = os.path.join(model_dir, 'GoogleNet_model_ep140')
    elif model_type == 'VGG':
        model = models.vgg16(pretrained=False, num_classes=1000)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                         2, bias=True)
        last_conv_layer = model.features[28]
        # Load checkpoint
        model_path = os.path.join(model_dir, 'VGG_model_ep140')
    # print(model_path)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, last_conv_layer


model_type = 'SqueezeNet'
model, last_conv_layer = load_model(model_type)
model = model.to(device).eval()
# Input preprocessing transformation
preprocessing = transforms.ToTensor()
normalize = transforms.Normalize((0.5, ), (0.5, ))

img = Image.open('../test_images/infected.jpg')
preproc_img = preprocessing(img)
plt.imshow(preproc_img.permute(1, 2, 0))
plt.axis('off')
plt.show()

# Normalize image and compute segmentation output
normalized_inp = normalize(preproc_img).unsqueeze(0).to(device)
normalized_inp.requires_grad = True
# out = model(normalized_inp)['out']
out = model(normalized_inp)
print(f'Predictd probability: {out}')

out_max = torch.argmax(out, dim=1, keepdim=True)
print(f'Predictd class: {out_max[0].cpu().detach().item()}')

lgc = LayerGradCam(model, last_conv_layer)
gc_attr = lgc.attribute(
    normalized_inp, target=out_max[0].cpu().detach().item())

# We can first confirm that the Layer GradCAM attributions match the dimensions of the layer activations.
# We can obtain the intermediate layer activation using the LayerActivation attribution method
la = LayerActivation(model, last_conv_layer)
activation = la.attribute(normalized_inp)
print("Input Shape:", normalized_inp.shape)
print("Layer Activation Shape:", activation.shape)
print("Layer GradCAM Shape:", gc_attr.shape)

"""
Comments from the source code
sign (string, optional): Chosen sign of attributions to visualize. Supported
            options are:
            1. `positive` - Displays only positive pixel attributions.
            2. `absolute_value` - Displays absolute value of
                attributions.
            3. `negative` - Displays only negative pixel attributions.
            4. `all` - Displays both positive and negative attribution
                values. This is not supported for `masked_image` or
                `alpha_scaling` modes, since signed information cannot
                be represented in these modes.
            Default: `absolute_value`
"""


viz.visualize_image_attr(gc_attr[0].cpu().permute(
    1, 2, 0).detach().numpy(), sign="all", show_colorbar=True)

upsampled_gc_attr = LayerAttribution.interpolate(
    gc_attr, normalized_inp.shape[2:])
print("Upsampled Shape:", upsampled_gc_attr.shape)

viz.visualize_image_attr_multiple(upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                  original_image=preproc_img.permute(
                                      1, 2, 0).numpy(),
                                  signs=["all", "positive",
                                         "negative", "absolute_value"],
                                  methods=[
                                      "original_image", "blended_heat_map", "blended_heat_map", "blended_heat_map"],
                                  show_colorbar=True)
