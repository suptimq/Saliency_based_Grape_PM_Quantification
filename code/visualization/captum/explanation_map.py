import torch
import torch.nn as nn

import numpy as np

from scipy.ndimage.interpolation import zoom

from visualization.captum.dataloader import HyphalDataset


np.random.seed(1701)
torch.manual_seed(1701)


class EM:
    r"""
    Computeds Explanation Map attribution for chosen layer.
    Reference based attention map generating algorithms proposed in
    An explainable deep machine vision framework for plant stress phenotyping
    Sambuddha Ghosala,1, David Blystoneb,1, Asheesh K. Singhb, Baskar Ganapathysubramaniana, Arti Singhb,2, and Soumik Sarkara,2
    input:
        target_image_features : target image feed forward activation featuers
        regerence_images_features : reference_image(s) feed forward activation features.
    returns: heatmap

    Examples::

        dataset_path = {
            'root_path': opt.dataset_path,
            'meta_filepath': 'metadata.csv',
            'train_filepath': 'train_set.hdf5',
            'test_filepath': 'test_set.hdf5'
        }
        net = ImageClassifier()
        explanation_map = EM(net, dataset_path, 0, 'features.0', transform=ToTensor())
        input = torch.randn(1, 3, 224, 224, requires_grad=True)
        attr = explanation_map.attribute(input)
    """

    def __init__(
        self,
        model,
        dataset_path,
        ref_class,
        layername,
        image_width=224,
        batch_size=64,
        transform=None,
        device='cpu'
    ):
        self.model = model
        self.reference_images_ds = HyphalDataset(
            dataset_path, target_class=ref_class, train=False, transform=transform)
        self.reference_images_dl = torch.utils.data.DataLoader(
            self.reference_images_ds, batch_size=batch_size, shuffle=False)

        # self.layername_list = [layername] if layername else self._get_layername_list()
        self.layername = layername

        self.features_blob = []
        self.hook_hanlder = []
        self._register_hook(self.model, self.layername)

        self.image_width = image_width
        self.device = device

        self.ASA = self._ASA()

    def attribute(
        self,
        inputs
    ):
        self.features_blob.clear()
        _ = self.model(inputs)
        target_image_features = self.features_blob[0]
        self.features_blob.clear()

        em = self._EM(target_image_features, self.ASA)

        return em

    def _get_layername_list(self):
        model_name = self.model.__class__.__name__
        layername_list = []
        if model_name == 'VGG':
            for idx, layer in self.model.features._modules.items():
                if isinstance(layer, nn.Conv2d):
                    layername_list.append(f'features.{idx}')

            return layername_list

    def _get_features_hook(
        self,
        module,
        input,
        output
    ):
        self.features_blob.append(output.cpu().detach().numpy())

    def _register_hook(self, model, layername):
        for (name, module) in self.model.named_modules():
            if name == layername:
                self.hook_hanlder.append(
                    module.register_forward_hook(self._get_features_hook))

    def _calculate_feature_blobs(
        self,
        model,
        imageloader
    ):
        self.features_blob.clear()

        for images, _ in imageloader:
            _ = self.model(images.to(self.device).float())

        reference_features = np.concatenate(self.features_blob, axis=0)

        self.features_blob.clear()

        return reference_features


    def _ASA(
        self
    ):
        reference_images_features = self._calculate_feature_blobs(
            self.model, self.reference_images_dl)

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

        ASA = np.mean(ASA) + 3 * np.std(ASA)  # the threshold
        #print("SA threshold of the given reference images is:",ASA)

        return ASA

    def _EM(
        self, 
        target_image_features, 
        ASA
    ):
        # intermediate output of interest of img
        Auv = target_image_features

        FI = []
        for i in range(target_image_features.shape[1]):  # per channel
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

        EMuvs = zoom(EMuvs, self.image_width/EMuvs.shape[0])

        return EMuvs