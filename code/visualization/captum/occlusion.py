import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as tvtrans

from captum.attr import Occlusion

# Default device
device = "cuda:1" if torch.cuda.is_available() else "cpu"


def VisualizeImageGrayscale(image_3d, percentile=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def abs_grayscale_norm(img):
    """Returns absolute value normalized image 2D."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"

    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        img = np.absolute(img)
        img = img/float(img.max())
    else:
        img = VisualizeImageGrayscale(img)
    return img


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


model, last_conv_layer = load_model()
model = model.to(device).eval()
# Input preprocessing transformation
preprocess = tvtrans.Compose([
    tvtrans.ToPILImage(),
    tvtrans.ToTensor(),
    tvtrans.Normalize((0.5, ), (0.5, ))
])
image_width = image_height = 224

img = Image.open('../test_images/tray1_139-4510009_Vertical_patch_14.png')
img_arr = np.asarray(img)
input_img = preprocess(img_arr).unsqueeze(0).to(device)
input_img.requires_grad = True


logits = model(input_img)
logtis_class = torch.argmax(logits, axis=1)[0].cpu().detach().item()
print(f'Predictd probability: {logtis_class}')

occlusion = Occlusion(model)
strides = (3, 1, 1)
# sliding_window_shapes = [(3, 15, 15), (3, 12, 12), (3, 9, 9), (3, 6, 6)]
sliding_window_shapes = [(3, 3, 3)]

output_masks = {}
count = 0
for sliding_window_shape in sliding_window_shapes:
    attr = occlusion.attribute(input_img, target=logtis_class,
                               strides=strides, sliding_window_shapes=sliding_window_shape)

    if len(attr.shape) > 3:
        attr = attr[0]

    if not isinstance(attr, np.ndarray):
        attr = attr.cpu().detach().numpy()

    key = f'{sliding_window_shape[1]}-{sliding_window_shape[2]}'
    output_masks[key] = abs_grayscale_norm(attr.transpose(1, 2, 0))
    count += 1

titles = list(output_masks.keys())
fig, axs = plt.subplots(3, 3, figsize=(11, 11))
finish = False
count = 0

for i in range(3):
    for j in range(3):
        heat_map = axs[i, j].imshow(
            output_masks[titles[count]], vmin=-1.0, vmax=1.0, cmap='bwr')
        axs[i, j].set_title(str(count))
        axs[i, j].set_xticklabels([])
        axs[i, j].set_yticklabels([])

        count += 1
        if count == len(list(output_masks.keys())):
            finish = True
            break

    if finish:
        break

fig.suptitle(f'Occlusion with fixed stride {strides}')
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
fig.colorbar(heat_map, orientation="horizontal", cax=cbar_ax)

plt.savefig('occlusion.png')
