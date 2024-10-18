import torch
import torch.nn as nn
import os
import sys

from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, './DeepLabV3Plus-Pytorch')

import network

from datasets import Cityscapes

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

img_path = "images/test.jpg"

device = "cpu"

img = Image.open(img_path).convert('RGB')
img = transform(img).unsqueeze(0)  # To tensor of NCHW
img = img.to(device)

model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19,
                                              output_stride=16)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

set_bn_momentum(model.backbone, momentum=0.01)

weights = "weights/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
checkpoint = torch.load(weights, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model = model.eval()

pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
colorized_preds = Cityscapes.decode_target(pred).astype('uint8')
colorized_preds = Image.fromarray(colorized_preds)

colorized_preds.save(
    os.path.join('images/result_torch.png'))
