import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Union
import numpy as np
from sklearn.decomposition import PCA
class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, image: Union[str, Image.Image]) -> np.ndarray:
        # Preprocess the input image
        # Load image ONLY if a path is provided 
        if isinstance(image, str):  
            input_image = Image.open(image).convert("RGB") 
        elif isinstance(image, Image.Image):
            input_image = image.convert("RGB")  # Use the provided Image directly
        else:
            raise TypeError("Input 'image' must be a file path (str) or a PIL Image object.")
        input_image = self.preprocess(input_image)
        input_tensor = input_image.unsqueeze(0)  # Add batch dimension her
        with torch.no_grad():
            output = self.model(input_tensor)

        # Flatten the output tensor, preserving the batch dimension
        features = output.squeeze().numpy()  
        # Normalize and flatten the feature vector
        return normalize(features.reshape(1, -1), norm="l2").flatten()