# BlackBird

This project aims to provide necessary modules for images collected using PM/BlackBird robots. The idea is to develop Python-based image processing pipeline including Image Analysis and Result Visualization modules for the phenotyping facility services.

### Requirements

- [Singularity >= 2.6.1](https://sylabs.io/guides/3.2/user-guide/index.html)
- [Python 3.6](https://www.python.org/)
- [PyTorch >= 1.4](https://pytorch.org/)
- [TorchVision](https://pypi.org/project/torchvision/)
- [Captum](https://github.com/pytorch/captum)

There is a pretty good [tutorial](https://github.com/bdusell/singularity-tutorial) you can follow to get a taste for Singularity

#### Build

> **_NOTE:_**: You must be the root user to build from a Singularity recipe file

```Python
singularity build Ubuntu18-CUDA10-cudnn7.simg Ubuntu18-CUDA10-cudnn7.def
```

#### Install Python Package

Since we are separating Python modules from the image using [Pipenv](https://github.com/pypa/pipenv), there is one more effort to be made to install the required Python libraries. Make sure you are in the directory where the _Pipfile_ is then run the following command:

```Python
singularity exec Ubuntu18-CUDA10-cudnn7.simg pipenv install
```

Or you might want to install more packages

```Python
singularity exec Ubuntu18-CUDA10-cudnn7.simg pipenv install torch numpy opencv-python matplotlib (etc..)
```

#### Test

Run the following command to test the environment

```Python
singularity exec --nv Ubuntu18-CUDA10-cudnn7.simg pipenv run python3 test.py
```

#### Run

After setup the running environment, there are many ways to execute the code. What we choose is as follows:

```Python
Singularity shell --nv Ubuntu18-CUDA10-cudnn7.simg
Singularity: Invoking an interactive shell within container...

Singularity Ubuntu18-CUDA10-cudnn7.simg:~/{path}>sh /.../code/run.sh
```

> **_NOTE:_**: If the dataset is out of the code direcotry, for example, the code directory is _/home/user1_ while the dataset is on the _/media/dataset1_, you need to use **[bind](https://sylabs.io/guides/3.5/user-guide/quick_start.html#working-with-files)** flag when running singularity

### Structure

The basic filesystem structure is like what the following tree displays. Essentially, we have two folders **code** and **data** to store scripts and dataset, respectively.

```console
BB_analysis/
├── code
│   ├── analysis
│   ├── classification
│   ├── common
│   ├── converter
│   ├── scripts
│   ├── visualization
│   ├── __init__.py
│   ├── analyzer.py
│   ├── analyzer_patch.py
│   └── analyzer_severity.py
├── data
│   └── Hyphal_2019
│       ├── metadata.csv
│       ├── test_set.hdf5
│       ├── train_set.hdf5
│       └── images
├── Pipfile
├── README.md
├── results
│   ├── logs
│   ├── models
│   └── runs
├── test.py
└── Ubuntu18-CUDA10-cudnn7.def
```