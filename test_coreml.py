import os
import sys
import coremltools
import numpy as np
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, './DeepLabV3Plus-Pytorch')

from datasets.cityscapes import Cityscapes

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    T.ToPILImage()
])
resize = (2048, 1024)


def preprocess_image(img):
    desired_aspect = resize[0] / resize[1]
    original_width, original_height = img.size
    original_aspect = original_width / original_height

    if original_aspect > desired_aspect:
        new_height = original_height
        new_width = int(original_height * desired_aspect)
        left = (original_width - new_width) // 2
        top = 0
    else:
        # Crop the height
        new_width = original_width
        new_height = int(original_width / desired_aspect)
        left = 0
        top = (original_height - new_height) // 2

    right = left + new_width
    bottom = top + new_height
    im_cropped = img.crop((left, top, right, bottom))

    im_cropped = im_cropped.resize(resize, Image.BILINEAR)
    im_cropped = transform(im_cropped)

    return im_cropped


model = coremltools.models.MLModel(f"weights/deeplabv3plus_mobilenet.mlmodel")

image = Image.open("images/test.jpg")
image = preprocess_image(image)

# Predict using CoreML
coreml_output = model.predict({"input": image})

# Extract the output
coreml_output = coreml_output["output"]

# Convert to numpy array
coreml_output = np.array(coreml_output)

colorized_preds = Cityscapes.decode_target(coreml_output).astype('uint8')
colorized_preds = Image.fromarray(colorized_preds)

colorized_preds.save(os.path.join('images/result_coreml.png'))