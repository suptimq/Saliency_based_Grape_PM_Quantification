## Visualization Component

This is the visualization component for PM/BlackBird robots. Essentially, we would like to understand the decision-making of the models by employing state-of-the-art visualization algorithms.

### References

The first listed repository contains a number of CNN visualization techniques. The drawback is that it only tests these techniques with AlexNet so there might be potential problems when using other models like ResNet. The second one supports more torchvision models with several layer-finding functions in its **utils.py**.

- [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
- [gradcam_plus_plus-pytorch](https://github.com/1Konny/gradcam_plus_plus-pytorch)

### Papers

- [Guided-Backpropagation](https://arxiv.org/pdf/1412.6806.pdf)
- [CAM](https://arxiv.org/pdf/1512.04150.pdf)
- [Grad-CAM](https://arxiv.org/pdf/1610.02391.pdf)

### Files

All the files can be ran directly through `python` command. They are also imported in the analysis module except **cam.py**.

```console
- cam.py: The implementation of CAM
- gradcam.py: The implementation of Grad-CAM
- guided_backprop.py: The implementation of Guided-Backpropagation
- guided_backprop_resnet.py: The Guided-BP for ResNet
- guided_gradcam.py: The combination of Grad-CAM and Guided-BP
```

### Issues

- []CUDA compatible
- []Batch compatible

Currently, there are two issues to be resolved. The **guided_backprop.py** works weirdly when using CUDA, the issues occurs in line **37**:

```Python
def hook_layers(self):
    def hook_function(module, grad_in, grad_out):
        self.gradients = grad_in[0]
        # pdb.set_trace()
    # Register hook to the first layer
    first_layer = list(self.model.features._modules.items())[0][1]
    first_layer.register_backward_hook(hook_function)
```

The **gradcam.py** is not compatible with batch inference. We believe the modification should be done in the `__cal__` fucntion of the `GradCam` class:

```Python
class GradCam:
    def __init__(...):
        ...
    def __call__(self, input_image, index=None):
        ...

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        ...

        return cam
```