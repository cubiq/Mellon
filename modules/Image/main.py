from mellon.NodeBase import NodeBase
from PIL import Image
from mellon.config import CONFIG
from pathlib import Path
import logging
from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
logger = logging.getLogger('mellon')

class Load(NodeBase):
    """
    Load an image from a file
    """

    label = "Load Image"
    category = "image"
    resizable = True
    params = {
        "file": {
            "label": "File",
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {
                "fileTypes": ["image"],
            },
        },
        "image": {
            "label": "Image",
            "display": "output",
            "type": "image",
        },
        "width": { "display": "output", "type": "int" },
        "height": { "display": "output", "type": "int" },
    }

    def execute(self, **kwargs):
        file = kwargs["file"]
        
        # The file browser element returns a list of files. TODO: handle multiple files
        if isinstance(file, list):
            file = file[0]

        if not Path(file).is_absolute():
            file = Path(CONFIG.paths['work_dir']) / file

        if not Path(file).exists():
            logger.error(f"File {file} not found")
            return {"image": None, "width": None, "height": None}

        # load image
        try:
            image = Image.open(file)
        except Exception as e:
            logger.error(f"Error loading image {file}: {e}")
            return {"image": None, "width": None, "height": None}
        
        return {"image": image, "width": image.width, "height": image.height}

class Preview(NodeBase):
    """
    Preview an image
    """
    label = "Preview Image"
    category = "image"
    resizable = True
    params = {
        "vae": { "type": "pipeline", "display": "input", "label": "VAE" },
        "device": { "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
        "image": { "type": ["image", "latent"], "display": "input", "onChange": {
            "action": "show",
            "data": { True: ["vae", "device"], False: [] },
            "condition": { "type": "latent" }
        }},
        "preview": { "display": "ui_image", "type": "url", "dataSource": "output" },
        "output": { "type": "image", "display": "output" },
    }
    
    def execute(self, **kwargs):
        image = kwargs["image"]
        if image is None:
            return {"output": None}
        
        # if image is an Image or an array of Images, pass it to the preview
        if isinstance(image, Image.Image) or (isinstance(image, list) and len(image) > 0 and isinstance(image[0], Image.Image)):
            return {"output": image}

        from modules.Experiments.VAE import VAEDecode
        pipeline = kwargs["vae"]
        device = kwargs["device"]
        if pipeline is None:
            logger.error("VAE is required to decode latents")
            return {"output": None}
        pipeline = pipeline.vae if hasattr(pipeline, 'vae') else pipeline
        vae = VAEDecode()
        if isinstance(image, list) or isinstance(image, tuple):
            output = self.mm_exec(lambda: vae.decode(pipeline, image[0], image[1]), device, models=[pipeline])
        else:
            output = self.mm_exec(lambda: vae.decode(pipeline, image), device, models=[pipeline])

        return {"output": output}


class Resize(NodeBase):
    """
    Resize an image or a list of images
    """

    label = "Resize"
    category = "image"
    params = {
        "image": { "type": "image", "display": "input" },
        "method": { "type": "string", "default": "stretch", "options": ["stretch", "cover", "contain", "fit", "pad"],
                   "description": """
                   The method to use to resize the image.
                   `Stretch` will stretch the image to the given width and height.
                   `Cover` will resize to the maximum dimension that fits the given size keeping the aspect ratio.
                   `Contain` will resize to the minimum dimension that fits the given size keeping the aspect ratio.
                   `Fit` will resize to the given size and crop the excess.
                   `Pad` will resize to the given size and pad the excess space with black.
                    """
        },
        "width": { "type": "int", "default": 1024 },
        "height": { "type": "int", "default": 1024 },
        "multiple_of": { "label": "Multiple of", "type": "int", "default": 1, "min": 1, "description": "Ensure the resulting width and height are multiples of this value." },
        "interpolation": { "default": "bicubic", "options": ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"] },
        "output": { "label": "Image", "type": "image", "display": "output" },
    }
    
    def execute(self, **kwargs):
        from utils.image import resize, fit, cover, contain, pad
        
        image = kwargs["image"]
        method = kwargs["method"]
        width = kwargs["width"]
        height = kwargs["height"]
        interpolation = kwargs["interpolation"]
        multiple_of = kwargs["multiple_of"]

        if multiple_of > 1:
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of

        if method == "stretch":
            image = resize(image, width, height, interpolation)
        elif method == "cover":
            image = cover(image, width, height, interpolation)
        elif method == "contain":
            image = contain(image, width, height, interpolation)
        elif method == "fit":
            image = fit(image, width, height, interpolation)
        elif method == "pad":
            image = pad(image, width, height, interpolation)
        
        return {"output": image}

