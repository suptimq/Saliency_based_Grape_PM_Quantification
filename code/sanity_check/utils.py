import os
import sys
import PIL
import h5py
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from captum.attr import (GuidedGradCam, GuidedBackprop, IntegratedGradients, InputXGradient, Occlusion, DeepLift,
                         NoiseTunnel, LayerGradCam, LayerActivation, LayerAttribution, Saliency)

sys.path.append(os.path.dirname(sys.path[0]))

from visualization.captum.explanation_map import EM
from classification.inception3 import inception_v3
from classification.resnet50 import resnet50

np.random.seed(1701)
torch.manual_seed(1701)


def printArgs(logger, args):
    for k, v in args.items():
        if logger:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


def load_f5py(dataset_para):
    """
        Load data from HDF5 files or image directory
    """
    f = h5py.File(dataset_para['root_path'] /
                  dataset_para['test_filepath'], 'r')
    image_ds = f['images']
    images = image_ds[:, ]
    label_ds = f['labels']
    labels = label_ds[:]
    return images, labels


def parse_model(opt):
    result_root_path = Path(opt.model_path) / 'results'

    current_time = opt.timestamp
    model_type_time = opt.model_type + '_{}'.format(current_time)

    model_path = result_root_path / 'models' / model_type_time
    model_filename = '{0}_model_ep{1:03}'

    cuda_id = opt.cuda_id if opt.cuda and torch.cuda.is_available() else None
    model_para = {
        'model_type': opt.model_type,
        'pretrained': opt.pretrained,
        'outdim': opt.outdim,
        'model_path': model_path,
        'model_filename': model_filename,
        'loading_epoch': opt.loading_epoch,
        'cuda': opt.cuda,
        'cuda_id': cuda_id
    }

    return model_para


def load_model(model_para):
    """
        Load well-trained model
    """
    model = init_model(model_para)
    model_fullpath = str(
        model_para['model_path'] / model_para['model_filename'])

    cuda_id = model_para['cuda_id']
    device = torch.device(
        f'cuda:{cuda_id}' if cuda_id else 'cpu')

    checkpoint = torch.load(model_fullpath.format(
        model_para['model_type'], model_para['loading_epoch']), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.to(device), device


def init_model(model):
    m = None
    outdim = model['outdim']
    pretrained = model.get('pretrained', False)
    if model['model_type'] == 'GoogleNet':
        m = models.googlenet(pretrained=pretrained, num_classes=1000)
        m.fc = nn.Linear(m.fc.in_features, outdim, bias=True)
        if not pretrained:
            m.aux1.fc2 = nn.Linear(m.aux1.fc2.in_features, outdim, bias=True)
            m.aux2.fc2 = nn.Linear(m.aux2.fc2.in_features, outdim, bias=True)

    elif model['model_type'] == 'SqueezeNet':
        m = models.squeezenet1_1(pretrained=pretrained, num_classes=1000)
        m.classifier[1] = nn.Conv2d(m.classifier[1].in_channels,
                                    outdim,
                                    kernel_size=(1, 1),
                                    stride=(1, 1))
    elif model['model_type'] == 'ResNet':
        m = resnet50(pretrained=pretrained, num_classes=1000)
        m.fc = nn.Linear(m.fc.in_features, outdim, bias=True)
    elif model['model_type'] == 'DenseNet':
        m = models.densenet161(pretrained=pretrained, num_classes=1000)
        m.classifier = nn.Linear(m.classifier.in_features,
                                 outdim,
                                 bias=True)
    elif model['model_type'] == 'AlexNet':
        m = models.alexnet(pretrained=pretrained, num_classes=1000)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features,
                                     outdim, bias=True)
    elif model['model_type'] == 'VGG':
        m = models.vgg16(pretrained=pretrained, num_classes=1000)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features,
                                     outdim, bias=True)
    elif model['model_type'] == 'Inception3':
        m = inception_v3(pretrained=pretrained, num_classes=1000)
        m.fc = nn.Linear(m.fc.in_features, outdim, bias=True)
        m.AuxLogits.fc = nn.Linear(
            m.AuxLogits.fc.in_features, outdim, bias=True)

    assert m != None, 'Model Not Initialized'
    return m


def VisualizeImageGrayscale(image_3d, percentile=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def VisualizeImageDiverging(image_3d, percentile=99):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
    """
    image_2d = np.sum(image_3d, axis=2)

    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)


def LoadImage(file_path, resize=True, sztple=(224, 224)):
    img = PIL.Image.open(file_path).convert('RGB')
    if resize:
        img = img.resize(sztple, PIL.Image.ANTIALIAS)
    img = np.asarray(img)
    return img / 127.5 - 1.0


def ShowImage(im, title='', ax=None):
    if ax is None:
        plt.figure()
    plt.axis('off')
    im = ((im + 1) * 127.5).astype(np.uint8)
    plt.imshow(im)
    plt.title(title)


def normalize_image(x):
    x = np.array(x).astype(np.float32)
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


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


def diverging_norm(img):
    """Returns image with positive and negative values."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"

    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        imgmax = np.absolute(img).max()
        img = img/float(imgmax)
    else:
        img = VisualizeImageDiverging(img)
    return img


def attribute_image_features(net, algorithm, input, target, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=target,
                                              **kwargs
                                              )

    return tensor_attributions


def init_weights(m):
    # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
        # nn.init.uniform_(m.weight.data)
        # if m.bias is not None:
        #     nn.init.uniform_(m.bias.data)


def reinit(model, blocklist):
    for block in blocklist:
        block_split = block.split('/')
        if len(block_split) > 1:
            block = block_split[0]
            layer = block_split[1]
            module = model._modules.get(block)[int(layer)]
        else:
            block = block_split[0]
            module = model._modules.get(block)
        module.apply(init_weights)


def get_last_conv_layer(model):
    """
        Get the last convolutional layer given model
    """
    model_name = model.__class__.__name__
    if model_name == 'VGG':
        last_conv_layer = model.features[28]
    elif model_name == 'SqueezeNet':
        last_conv_layer = model.features[12].expand3x3
    elif model_name == 'ResNet':
        last_conv_layer = model.layer4[-1].conv2
    elif model_name == 'Inception3':
        last_conv_layer = model.Mixed_7c

    return last_conv_layer


def get_layer_randomization_order(model, mode):
    """
        Get the layer order given model
    Args:
        model:       Specified nn models
        mode:        Randomization mode (cascading or independent)
    """
    model_name = model.__class__.__name__
    if model_name == 'VGG':
        if mode == 'cascading':
            # return ['classifier', 'features/28', 'features/26', 'features/24',
            #         'features/21', 'features/19', 'features/17', 'features/14', 'features/12', 'features/10',
            #         'features/7', 'features/5', 'features/2', 'features/0']
            return ['classifier', 'features/28', 'features/26', 'features/21', 'features/14', 'features/0']
        else:
            # return ['classifier', 'features/24', 'features/17', 'features/0']
            return ['classifier', 'features/24']
    elif model_name == 'ResNet':
        return ['fc', 'layer4', 'layer3', 'layer2',
                'layer1', 'conv1']
    elif model_name == 'Inception3':
        if mode == 'cascading':
            return ['fc', 'Mixed_7c',
                    'Mixed_7b', 'Mixed_7a',
                    'Mixed_6e', 'Mixed_6d',
                    'Mixed_6c', 'Mixed_6b',
                    'Mixed_6a', 'Mixed_5d',
                    'Mixed_5c', 'Mixed_5b',
                    'Conv2d_4a_3x3', 'Conv2d_3b_1x1',
                    'Conv2d_2b_3x3', 'Conv2d_2a_3x3',
                    'Conv2d_1a_3x3']
            # return ['fc', 'Mixed_7c', 'Mixed_7b', 'Mixed_6e', 'Mixed_5b', 'Conv2d_1a_3x3']
            # return ['fc', 'Mixed_7c', 'Conv2d_1a_3x3']
        else:
            return ['fc', 'Mixed_6d', 'Mixed_5b', 'Conv2d_1a_3x3']


def get_first_conv_layer(model):
    """
        Get the first convolution layer given model
    """
    model_name = model.__class__.__name__
    if model_name == 'VGG':
        return 'features.0'
    elif model_name == 'ResNet':
        return 'conv1'
    elif model_name == 'Inception3':
        return 'Conv2d_1a_3x3'


def get_saliency_methods(model, **kwargs):
    """
        Get saliency methods
    Args:
        methods:     List of saliency methods
    """
    last_conv_layer = kwargs.get('last_conv_layer', None)
    first_conv_layer = kwargs.get('first_conv_layer', None)
    ref_dataset_path = kwargs.get('ref_dataset_path', None)
    image_width = kwargs.get('image_width', None)
    transform = kwargs.get('transform', None)
    device = kwargs.get('device', None)

    assert last_conv_layer, 'last_conv_layer not found'
    assert first_conv_layer, 'first_conv_layer not found'
    assert ref_dataset_path, 'ref_dataset_path not found'
    assert image_width, 'image_width not found'
    assert transform, 'transform not found'
    assert device, 'device not found'

    partial = kwargs.get('partial', False)

    if partial:
        gradient = kwargs.get('gradient', False)
        smooth_grad = kwargs.get('smooth_grad', False)
        gradcam = kwargs.get('gradcam', False)
        guided_backprop = kwargs.get('guided_backprop', False)
        integrated_gradients = kwargs.get('integrated_gradients', False)
        integrated_smooth_grad = kwargs.get('integrated_smooth_grad', False)
        input_grad = kwargs.get('input_grad', False)
        occlusion = kwargs.get('occlusion', False)
        deeplift = kwargs.get('deeplift', False)
        explanation_map = kwargs.get('explanation_map', False)
    else:
        gradient = smooth_grad = gradcam = guided_backprop = integrated_gradients = integrated_smooth_grad = occlusion = input_grad = deeplift = explanation_map = True

    saliency_methods = {}

    if gradient:
        saliency_methods['Gradient'] = Saliency(model)
    if smooth_grad:
        saliency_methods['Gradient-SG'] = NoiseTunnel(Saliency(model))
    if gradcam:
        saliency_methods['GradCAM'] = LayerGradCam(model, last_conv_layer)
    if deeplift:
        saliency_methods['DeepLift'] = DeepLift(model)
    if integrated_gradients:
        saliency_methods['Integrated\nGradients'] = IntegratedGradients(model)
    if integrated_smooth_grad:
        saliency_methods['IG-SG'] = NoiseTunnel(
            IntegratedGradients(model))
    if input_grad:
        saliency_methods['Input-Grad'] = InputXGradient(model)
    if occlusion:
        saliency_methods['Occlusion'] = Occlusion(model)
    if explanation_map:
        saliency_methods['Explanation\nMap'] = EM(
            model,
            ref_dataset_path,
            ref_class=0,
            layername=first_conv_layer,
            image_width=image_width,
            transform=transform,
            batch_size=32,
            device=device)
    if guided_backprop:
        saliency_methods['Guided\nBackProp'] = GuidedBackprop(model)

    return saliency_methods


def get_saliency_masks(saliency_methods, input_img, logits_class, relu_attributions=True):
    output_masks = {}
    for key, method in saliency_methods.items():
        if key == 'Explanation\nMap':
            attr = method.attribute(input_img)
            attr = attr[np.newaxis, ...]
        elif key == 'Occlusion':
            attr = method.attribute(input_img, target=logits_class, strides=(
                3, 8, 8), sliding_window_shapes=(3, 10, 10))
        elif key == 'GradCAM':
            attr = method.attribute(
                input_img, target=logits_class, relu_attributions=relu_attributions)
            attr = LayerAttribution.interpolate(
                attr, input_img.shape[2:], interpolate_mode='bilinear')
        elif 'SG' in key:
            attr = method.attribute(
                input_img, n_samples=4, stdevs=0.15, nt_type='smoothgrad', target=logits_class)
        else:
            attr = method.attribute(input_img, target=logits_class)

        if len(attr.shape) > 3:
            attr = attr[0]
        if not isinstance(attr, np.ndarray):
            attr = attr.cpu().detach().numpy()
        output_masks[key] = attr.transpose(1, 2, 0)

    if 'Guided\nBackProp' in output_masks.keys():
        output_masks['GBP-GC'] = np.multiply(
            output_masks['Guided\nBackProp'], output_masks['GradCAM'])

    return output_masks
