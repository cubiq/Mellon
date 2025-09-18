from mellon.NodeBase import NodeBase
from PIL import Image
from mellon.config import CONFIG
from pathlib import Path
import logging
from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
import nanoid

logger = logging.getLogger('mellon')

class Load(NodeBase):
    """
    Load an image from a file
    """

    label = "Load Image"
    category = "image"
    resizable = True
    params = {
        "image": {
            "label": "Image",
            "display": "output",
            "type": "image",
        },
        'label': {
            "display": "ui_label",
            "value": "Load Image",
        },
        "file": {
            "label": False,
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {
                "fileTypes": ["image"]
            },
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

class Save(NodeBase):
    """
    Save an image to a file
    """

    label = "Save Image"
    category = "image"
    style = { "minWidth": 360 }
    resizable = True
    params = {
        "image": {
            "label": "Image",
            "display": "input",
            "type": "image",
        },
        "filename": {
            "label": "File",
            "type": "str",
            "default": "{PATH:images}/Mellon_{HASH:6}.webp",
        },
        "quality": {
            "label": "Quality",
            "type": "int",
            "display": "slider",
            "default": 100,
            "min": 1,
            "max": 100
        },
        "output": {
            "label": "Filename",
            "display": "output",
            "type": "str",
        }
    }

    def execute(self, **kwargs):
        from pathlib import Path
        from PIL import Image

        image = kwargs.get("image")
        if not isinstance(image, list):
            image = [image]

        filename = kwargs.get("filename", "{PATH:images}/Mellon_{HASH:6}.webp")
        extension = filename.split('.')[-1].lower()
        quality = kwargs.get("quality", 95)
        output = []

        for index, img in enumerate(image):
            parsed_filename = self._parse_filename(filename, index=index)
            parsed_filename = Path(parsed_filename)

            if not parsed_filename.is_absolute():
                parsed_filename = Path(CONFIG.paths['data']) / parsed_filename

            try:
                if not parsed_filename.parent.exists():
                    parsed_filename.parent.mkdir(parents=True)

                if extension in ['jpg', 'jpeg']:
                    #image = img.convert("RGB")
                    img.save(parsed_filename, quality=quality)
                elif extension == 'png':
                    #image = img.convert("RGBA")
                    img.save(parsed_filename)
                elif extension == 'webp':
                    #image = img.convert("RGBA")
                    img.save(parsed_filename, quality=quality)
                else:
                    img.save(parsed_filename)
                output.append(str(parsed_filename))
            except Exception as e:
                logger.error(f"Error saving image {parsed_filename}: {e}")

        if len(output) == 0:
            output = None
        if len(output) == 1:
            output = output[0]

        return { "output": output }

    def _parse_filename(self, filename, **kwargs):
        from datetime import datetime
        import re

        def hash_func(arg, **kwargs): return nanoid.generate(size=int(arg))
        def date_func(arg, **kwargs): return datetime.now().strftime(arg)
        def index_func(arg, **kwargs): return "{:0>{width}}".format(kwargs.get("index", 0), width=arg)
        def path_func(arg, **kwargs): return CONFIG.paths.get(str(arg).lower()) or 'work_dir'
        local_map = locals()

        def replace_match(match):
            key = match.group(1).lower()
            arg = match.group(2)
            func_name = f"{key}_func"
            if func_name in local_map:
                return str(local_map[func_name](arg, **kwargs))
            return match.group(0)

        filename = re.sub(r'\{(\w+):([^\}]+)\}', replace_match, filename)
        return filename


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


class Compare(NodeBase):
    label = "Image Compare"
    category = "image"
    resizable = True
    params = {
        "image_1": {
            "label": "Image From",
            "display": "input",
            "type": "image",
        },
        "image_2": {
            "label": "Image To",
            "display": "input",
            "type": "image",
        },
        "image_list": {
            "display": "output",
            "type": "image",
            "hidden": True
        },
        "compare": {
            "label": "Compare",
            "display": "ui_imagecompare",
            "type": "url",
            "dataSource": "image_list"
        },
    }

    def execute(self, **kwargs):
        image_1 = kwargs.get("image_1")
        image_2 = kwargs.get("image_2")
        image_1 = image_1[0] if isinstance(image_1, list) and len(image_1) > 0 else image_1
        image_2 = image_2[0] if isinstance(image_2, list) and len(image_2) > 0 else image_2

        return {
            "image_list": [image_1, image_2]
        }

class ApplyMask(NodeBase):
    label = "Apply Mask"
    category = "image"
    params = {
        "image": {
            "label": "Image",
            "display": "input",
            "type": "image",
        },
        "mask": {
            "label": "Mask",
            "display": "input",
            "type": "image",
        },
        "output": {
            "label": "Output",
            "display": "output",
            "type": "image",
        }
    }

    def execute(self, **kwargs):
        image = kwargs.get("image")
        mask = kwargs.get("mask")

        if image is None or mask is None:
            return {"output": None}

        image = [image] if not isinstance(image, list) else image
        mask = [mask] if not isinstance(mask, list) else mask

        if len(mask) < len(image):
            diff = len(image) - len(mask)
            mask = mask + [mask[-1]] * diff
        elif len(mask) > len(image):
            mask = mask[:len(image)]

        # Apply the mask to the image
        output = []
        for img, msk in zip(image, mask):
            img = img.convert("RGBA")
            msk = msk.convert("L")
            img.putalpha(msk)
            output.append(img)

        return {"output": output}