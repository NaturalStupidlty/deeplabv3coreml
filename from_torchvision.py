# https://apple.github.io/coremltools/docs-guides/source/pytorch-conversion-examples.html

import torch
import torch.nn as nn
import json
import coremltools

from torchvision.models.segmentation.deeplabv3 import (
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


class DeepLabV3(nn.Module):
    SUPPORTED_WEIGHTS = {
        "deeplabv3_resnet101": DeepLabV3_ResNet101_Weights,
        "deeplabv3_resnet50": DeepLabV3_ResNet50_Weights,
        "deeplabv3_mobilenet_v3_large": DeepLabV3_MobileNet_V3_Large_Weights,
    }

    def __init__(self, model: str = "deeplabv3_resnet101", device: torch.device = None):
        super(DeepLabV3, self).__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            torch.hub.load(
                "pytorch/vision:v0.6.0",
                model,
                weights=self.SUPPORTED_WEIGHTS[model].DEFAULT,
            )
            .eval()
            .to(self.device)
        )

    def forward(self, x):
        """
        Run the forward pass of the model on the input x
        :param x: Input tensor of shape (1, 3, H, W)
        :return: Output tensor of shape (1, 1, H, W) (instead of raw like (1, 21, H, W))
        """
        output = self.model(x)["out"]
        return torch.argmax(output, dim=1, keepdim=True)


model_name = "deeplabv3_resnet50"
traceable_model = DeepLabV3(model_name).eval()

input_batch = torch.rand(1, 3, 256, 256).to(traceable_model.device)
with torch.no_grad():
    trace = torch.jit.trace(traceable_model, input_batch)

mlmodel = coremltools.convert(
    trace,
    convert_to="neuralnetwork",
    inputs=[
        coremltools.ImageType(name="input", shape=input_batch.shape)
    ],
    outputs=[
        coremltools.ImageType(
            name="output",
            color_layout=coremltools.colorlayout.GRAYSCALE
        )
    ],
)

labels_json = {
    "labels": [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "board",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningTable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedPlant",
        "sheep",
        "sofa",
        "train",
        "tvOrMonitor",
    ]
}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata["com.apple.coreml.model.preview.params"] = json.dumps(
    labels_json
)

mlmodel.save(f"{model_name}.mlmodel")
