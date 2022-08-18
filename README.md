# BlackBird

This project provides the source code and data for the manuscript "High throughput saliency-based quantification of grape powdery mildew at the microscopic level for disease resistance breeding" published in Oxford Academic/Horticulture Research.

![](data/Oxford_HR_Cover.png)

### Requirements

- [Python 3.6](https://www.python.org/)
- [PyTorch >= 1.4](https://pytorch.org/)
- [TorchVision](https://pypi.org/project/torchvision/)
- [Captum](https://github.com/pytorch/captum)

### Structure

The basic filesystem structure is like what the following tree displays. Essentially, we have two folders **code** and **data** to store scripts and dataset, respectively.

```console
Saliency_based_Grape_PM_Quantification/
├── code
│   ├── analysis
│   │   └── workflow
│   ├── classification
│   ├── common
│   ├── figures
│   ├── metric
│   ├── sanity_check
│   ├── script
│   ├── segmentation
│   └── visualization
│       ├── captum
│       └── test_images
├── data
│   └── Hyphal_2019
│       └── images
└── results
    ├── leaf_correlation
    │   └── saliency_based
    └── patch_correlation
        └── saliency_based
```

More detail explaination of each component in **code** can be found [here](code/README.md)

### Data

The raw data used in this study can be accessed via this [link](https://figshare.com/s/b0aa141a40fae00aea41).


### Model

The pretrained model of VGG, ResNet, Inception3, and DeepLab can be accessed via this [link](https://figshare.com/articles/online_resource/Pretrained_Models/20507070).