import coremltools
import torch
import sys

sys.path.insert(0, './DeepLabV3Plus-Pytorch')

import network

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = momentum


class DeepLabV3(torch.nn.Module):
    SUPPORTED_MODELS = {
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    NUM_CLASSES = 19

    def __init__(
        self,
        model_name: str = "deeplabv3plus_mobilenet",
        device: torch.device = None
    ):
        super(DeepLabV3, self).__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.SUPPORTED_MODELS[model_name](
            num_classes=self.NUM_CLASSES,
            output_stride=16
        )
        checkpoint = torch.load(
            weights,
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state"])
        set_bn_momentum(self.model.backbone, momentum=0.01)
        self.model.eval()

    def forward(self, x):
        """
        Run the forward pass of the model on the input x
        :param x: Input tensor of shape (1, 3, H, W)
        :return: Output tensor of shape (1, 1, H, W) (instead of raw (1, NUM_CLASSES, H, W))
        """
        output = self.model(x)
        return torch.argmax(output, dim=1, keepdim=True)


model_name = 'deeplabv3plus_mobilenet'
weights = 'weights/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'

traceable_model = DeepLabV3(model_name, device="cpu").eval()

input_batch = torch.rand(1, 3, 1024, 2048).to(traceable_model.device)
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
mlmodel.save(f"weights/{model_name}.mlmodel")
