"""
Source code: https://github.com/totti0223/keraswhitebox
"""
import os
import PIL
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage.interpolation import zoom

import torch
import torch.nn as nn
import torchvision.transforms as tvtrans
from torchvision import models

from dataloader import HyphalDataset

np.random.seed(100)
torch.manual_seed(100)


features_blob = []
hook_hanlder = []


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
        m = models.resnet34(pretrained=pretrained, num_classes=1000)
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
        m = models.inception_v3(pretrained=pretrained, num_classes=1000)
        m.fc = nn.Linear(m.fc.in_features, outdim, bias=True)
        m.AuxLogits.fc = nn.Linear(
            m.AuxLogits.fc.in_features, outdim, bias=True)

    assert m != None, 'Model Not Initialized'
    return m


def printArgs(args, logger=None):
    for k, v in vars(args).items():
        if logger:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


def get_features_hook(module, input, output):
    features_blob.append(output.cpu().detach().numpy())


def register_hook(model, layername):
    for (name, module) in model.named_modules():
        if name == layername:
            hook_hanlder.append(
                module.register_forward_hook(get_features_hook))


def EM(target_image_features, reference_images_features, image_width=224):
    '''
    reference based attention map generating algorithms proposed in
    An explainable deep machine vision framework for plant stress phenotyping
    Sambuddha Ghosala,1, David Blystoneb,1, Asheesh K. Singhb, Baskar Ganapathysubramaniana, Arti Singhb,2, and Soumik Sarkara,2
    input:
        target_image_features : target image feed forward activation featuers
        regerence_images_features : reference_image(s) feed forward activation features.
    returns: heatmap
    '''

    # print(reference_images_features.shape) ----> batch, channel, height, width,

    '''
    threshold = []
    for img_features in reference_images_features:
        for i in range(reference_images_features.shape[1]):
            feature = img_features[:,:,i]
            means = np.mean(feature)         
            threshold.append(means)

    threshold = np.array(threshold)
    threshold = np.reshape(threshold,(reference_images_features.shape[0], reference_images_features.shape[1]))
    '''

    ASA = []  # stress activation threshold

    # iterate channel in batch,height,width,channel
    for k in range(reference_images_features.shape[1]):
        ASA_per_channel = []
        for i in range(reference_images_features.shape[0]):  # iterate batch
            ASA_per_channel.append(
                np.mean(reference_images_features[i, k, :, :]))
        ASA_per_channel = np.array(ASA_per_channel)
        ASA.append(np.mean(ASA_per_channel))

    ASA = np.array(ASA)
    ASA_copy = ASA.copy()

    ASA = np.mean(ASA) + 3 * np.std(ASA)  # the threshold
    #print("SA threshold of the given reference images is:",ASA)

    # intermediate output of interest of img
    Auv = target_image_features

    FI = []
    for i in range(reference_images_features.shape[1]):  # per channel
        deltaAuv = Auv[:, i, :, :] - ASA  # [i]
        # indicator function check the subtracted feature map whether its positive or negative per pixel
        Iuv = deltaAuv.copy()
        Iuv[Iuv <= 0] = 0
        Iuv[Iuv > 0] = 1
        if np.sum(Iuv) != 0:
            FeatureImportanceMetric = np.sum(Iuv*deltaAuv)/np.sum(Iuv)
        else:
            FeatureImportanceMetric = 0
        FI.append(FeatureImportanceMetric)

    FI = np.array(FI)  # final feature importance metric

    explanationperimage = []
    # get top3 feature indxs
    #print("Auv shape is ",Auv.shape)
    indxs = np.argsort(-FI)[:3]
    for i in indxs:
        deltaAuv = Auv[0, i, :, :]-ASA  # [i]
        Iuv = deltaAuv.copy()
        Iuv[Iuv <= 0] = 0
        Iuv[Iuv > 0] = 1
        if np.sum(Iuv) == 0:
            break
        FeatureImportanceMetric = np.sum(Iuv*deltaAuv) / np.sum(Iuv)
        explanationperimage.append(FeatureImportanceMetric*Iuv*deltaAuv)
    EMuv = np.array(explanationperimage)
    #print("EMuv shape is",EMuv.shape)

    EMuvs = np.zeros((Auv.shape[2], Auv.shape[3]))  # height and width

    for i in range(EMuv.shape[0]):
        EMuvs += EMuv[i]

    EMuvs = zoom(EMuvs, image_width/EMuvs.shape[0])

    return EMuvs, ASA_copy


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


def deprocess_image(x):
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def calculate_feature_blobs(model, imageloader, device='cpu'):
    features_blob.clear()

    for images, _ in imageloader:
        images = images.to(device)
        _ = model(images.float())

    reference_features = np.concatenate(features_blob, axis=0)

    features_blob.clear()

    return reference_features


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        default='SqueezeNet',
                        help='model used for training')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model parameters')
    parser.add_argument('--loading_epoch',
                        type=int,
                        required=True,
                        help='xth model loaded for inference')
    parser.add_argument('--timestamp', required=True, help='model timestamp')
    parser.add_argument('--outdim', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--cuda', action='store_true', help='enable cuda')
    parser.add_argument('--cuda_id',
                        default="0",
                        help='specify cuda id')
    parser.add_argument('--ref_class',
                        default="0",
                        type=int,
                        help='class heatmap')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='root path to the data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='root path to the model')
    opt = parser.parse_args()
    printArgs(opt)
    return opt


if __name__ == "__main__":

    opt = arg_parser()
    model_para = parse_model(opt)
    # Model
    model, device = load_model(model_para)
    model.eval()

    root_path = Path(opt.dataset_path)

    ref_dataset_path = {
        'root_path': root_path,
        'train_filepath': root_path / 'train_set.hdf5',
        'test_filepath': root_path / 'test_set.hdf5'
    }

    ref_class = opt.ref_class

    if opt.model_type == 'Inception3':
        preprocess = tvtrans.Compose([
            tvtrans.ToPILImage(),
            tvtrans.Resize(299),
            # tvtrans.CenterCrop(299),
            tvtrans.ToTensor(),
            tvtrans.Normalize((0.5, ), (0.5, ))
        ])
        image_width = image_height = 299
    else:
        preprocess = tvtrans.Compose([
            tvtrans.ToPILImage(),
            tvtrans.ToTensor(),
            tvtrans.Normalize((0.5, ), (0.5, ))
        ])
        image_width = image_height = 224

    # Load data
    target_images_ds = HyphalDataset(
        ref_dataset_path, target_class=1, train=False, transform=preprocess)
    reference_images_ds = HyphalDataset(
        ref_dataset_path, target_class=ref_class, train=False, transform=preprocess)

    target_images_dl = torch.utils.data.DataLoader(
        target_images_ds, batch_size=1, shuffle=False)
    reference_images_dl = torch.utils.data.DataLoader(
        reference_images_ds, batch_size=64, shuffle=False)

    dataset_path = root_path / '07-08-19_6dpi' / 'labelbox_data_03-21'

    tray = 'tray1'
    leaf_disk_image_filename = '52-4508058'
    patch_filenames = ['tray1_52-4508058_Horizon_patch_11.png', 'tray1_52-4508058_Horizon_patch_19.png', 'tray1_52-4508058_Horizon_patch_13.png']

    # first_conv_layername_list = [
    #     'features.1', 'features.3', 'features.6', 'features.8',
    #     'features.11', 'features.13', 'features.15', 'features.18',
    #     'features.20', 'features.22', 'features.25', 'features.27',
    #     'features.29',
    # ]
    first_conv_layername_list = [
        'features.0', 'features.2', 'features.5', 'features.7', 'features.10',
        'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
        'features.24', 'features.26', 'features.28',
    ]

    output_image_patch_folder = Path(
        os.getcwd()).parent / 'results' / 'sanity_check' / 'EM' / 'VGG'
    if not os.path.exists(output_image_patch_folder):
        os.makedirs(output_image_patch_folder, exist_ok=True)

    format_ = 'pdf'

    from matplotlib import gridspec

    for patch_filename in patch_filenames:
        image_filepath = dataset_path / tray / leaf_disk_image_filename / patch_filename

        img = PIL.Image.open(image_filepath).resize(
            (image_width, image_height))
        demo_img = np.asarray(img)
        input_img = preprocess(demo_img).unsqueeze(0).to(device)

        output_image_patch_folder_ = output_image_patch_folder / patch_filename
        if not os.path.exists(output_image_patch_folder_):
            os.makedirs(output_image_patch_folder_, exist_ok=True)

        # plt.imshow(demo_img)
        # plt.axis('off')
        # plt.savefig(output_image_patch_folder_ /
        #             f'raw_image.{format_}', format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close()

        plt.imsave(output_image_patch_folder_ / f'raw_image.{format_}', demo_img, format=format_, dpi=300)

        nrows = 2
        ncols = 7

        fig = plt.figure(figsize=(7, 2))
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.025, hspace=0.025)

        row = 0
        col = 0

        for i, first_conv_layername in enumerate(first_conv_layername_list):
            print(first_conv_layername)

            # Register hook
            register_hook(model, first_conv_layername)

            # Feed forward through the target image
            features_blob.clear()
            logits = model(input_img)
            logtis_class = torch.argmax(logits, axis=1)[
                0].cpu().detach().item()
            target_image_features = features_blob[0]
            print(f'target features shape {target_image_features.shape}')
            features_blob.clear()

            # Feed forward through reference images
            reference_images_features = calculate_feature_blobs(
                model, reference_images_dl, device)
            print(
                f'reference features shape {reference_images_features.shape}')

            em, ASA = EM(target_image_features, reference_images_features)

            print(f'Explanation map shape {em.shape}')

            ax = plt.subplot(gs[row, col])

            ax.imshow(demo_img)
            ax.imshow(abs_grayscale_norm(em), cmap="hot", alpha=0.65)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            col += 1
            if col == ncols:
                row += 1
                col = 0

            # plt.savefig(output_image_patch_folder_ / f'{first_conv_layername}.{format_}',
            #             format=format_, dpi=300, bbox_inches='tight', pad_inches=0)

            # axes[n_row, n_col].imshow(demo_img)
            # plt.imshow(deprocess_image(em), cmap="hot", alpha=0.5)
            # axes[n_row, n_col].imshow(abs_grayscale_norm(em), cmap="hot", alpha=0.5)
            # axes[n_row, n_col].get_xaxis().set_visible(False)
            # axes[n_row, n_col].get_yaxis().set_visible(False)
            # plt.title(first_conv_layername, fontsize=8)
            # n_col += 1
            # if n_col == cols:
            #     n_col = 0
            #     n_row += 1
            # plot_num += 1
            # plt.subplot(row, col, plot_num)

            # Plot probability distribution of healthy leaf activations
            # plt.hist(ASA)
            # plt.title(first_conv_layername, fontsize=8)
            # plt.axis("off")
            # plt.xlabel('Probability distribution of healthy leaf activations')
            # plt.ylabel('Probability')

            # Remove hook
            hook_hanlder[-1].remove()

            # Free memory
            del target_image_features
            del reference_images_features

        plt.tight_layout()
        plt.savefig(output_image_patch_folder_ / f'{first_conv_layername}.{format_}',
                    format=format_, dpi=300, bbox_inches='tight', pad_inches=0)

        # fig.suptitle(f'Prediction class {logtis_class} Ground-truth {1}')
        # fig.tight_layout()
        # plt.axis('off')
        # plt.savefig(f'explanation_map_{patch_filename}.png')
