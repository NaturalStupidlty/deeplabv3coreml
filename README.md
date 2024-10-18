## 1. Using deeplabv3+

```bash
git submodule add https://github.com/VainF/DeepLabV3Plus-Pytorch.git
git submodule update --init --recursive
```

weights:
1. [deeplabv3plus_mobilenet](https://www.dropbox.com/scl/fi/jo4nhw3h6lcg8t2ckarae/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?rlkey=7qnzapkshyofrgfa1ls7vot6j&e=3&dl=0)
2. [deeplabv3plus_resnet101](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing)


## 2. Using torchvision

Patch the torchvision hubconf.py file to include the new models

```bash
nano .cache/torch/hub/pytorch_vision_v0.6.0/hubconf.py
```

```python
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101, deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
```
