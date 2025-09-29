from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from mellon.NodeBase import NodeBase

class InSPyReNetRemover(NodeBase):
    label = "InSPyReNet Transparent Background"
    category = "segmentation"
    params = {
        "image": {
            "label": "Image",
            "display": "input",
            "type": "image",
        },
        "checkpoint": {
            "label": "Checkpoint",
            "type": "string",
            "options": ["base", "base-nightly", "fast"],
            "default": "base"
        },
        "hard_edges": {
            "label": "Hard Edges",
            "type": "boolean",
            "default": False,
            "onChange": {
                True: ["he_threshold"],
                False: []
            }
        },
        "he_threshold": {
            "label": "Hard Edges Threshold",
            "display": "slider",
            "type": "float",
            "default": 0.5,
            "step": 0.01,
            "min": 0.0,
            "max": 1
        },
        "prediction": {
            "label": "Prediction",
            "display": "output",
            "type": "image",
        },
        "mask": {
            "label": "Mask",
            "display": "output",
            "type": "image",
        },
    }

    def execute(self, **kwargs):
        from transparent_background import Remover

        image = kwargs.get("image")
        hard_edges = kwargs.get("hard_edges", False)
        he_threshold = kwargs.get("he_threshold", 0.5)
        threshold = None
        if hard_edges:
            threshold = he_threshold

        image = [image] if not isinstance(image, list) else image

        masks = []
        predictions = []
        for img in image:
            remover = Remover()

            if img.mode != 'RGB':
                img = img.convert("RGB")
            mask = remover.process(img, type='map', threshold=threshold)
            masks.append(mask)

            pred = img.convert("RGBA")
            pred.putalpha(mask.convert("L"))
            predictions.append(pred)

        return { "mask": masks, "prediction": predictions }

class RemBg(NodeBase):
    """
    Remove background from an image using Rembg
    """

    label = "Remove Background (Rembg)"
    category = "segmentation"
    params = {
        "image": {
            "label": "Image",
            "display": "input",
            "type": "image",
        },
        "prediction": {
            "label": "Prediction",
            "display": "output",
            "type": "image",
        },
        "mask": {
            "label": "Mask",
            "display": "output",
            "type": "image",
        },
    }

    def execute(self, **kwargs):
        from rembg import remove

        image = kwargs.get("image")
        image = [image] if not isinstance(image, list) else image

        predictions = []
        masks = []
        for img in image:
            if img.mode != 'RGB':
                img = img.convert("RGB")
            prediction = remove(img)
            if prediction.mode != 'RGBA':
                prediction = prediction.convert("RGBA")
            alpha = prediction.split()[-1]
            predictions.append(prediction)
            masks.append(alpha)

        return {"prediction": predictions, "mask": masks}
    
# class Sam2Auto(NodeBase):
#     label = "Segment Anything v2"
#     category = "segmentation"
#     params = {
#         "image": {
#             "label": "Image",
#             "display": "input",
#             "type": "image",
#         },
#         "masks": {
#             "label": "Masks",
#             "display": "output",
#             "type": "image",
#         },
#         "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
#     }

#     def execute(self, **kwargs):
#         import torch
#         from transformers import pipeline
#         image = kwargs.get("image")
#         image = image[0] if isinstance(image, list) else image
#         device = kwargs.get("device", DEFAULT_DEVICE)
#         generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=device)
#         output = generator(image, points_per_batch=64)
#         masks = output['masks']
#         # convert the binary tensors to PIL images
#         from PIL import Image
        
#         pil_masks = []
#         for mask_tensor in masks:
#             mask_numpy = (mask_tensor.squeeze().numpy() * 255).astype('uint8')
#             pil_mask = Image.fromarray(mask_numpy, 'L')
#             pil_masks.append(pil_mask)

#         return {"masks": pil_masks}