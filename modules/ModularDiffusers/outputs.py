from pathlib import Path

from diffusers.utils import load_image
from PIL import Image

from mellon.config import CONFIG
from mellon.NodeBase import NodeBase

from . import MESSAGE_DURATION


def flatten_images(images):
    image_list = []
    if isinstance(images, list):
        for item in images:
            image_list.extend(flatten_images(item))
    elif isinstance(images, Image.Image):
        image_list.append(images)
    return image_list


def convert_channels(image, channels):
    if channels == "RGB (3 channels)":
        if image.mode != "RGB":
            image = image.convert("RGB")
    elif channels == "RGBA (4 channels)":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
    return image


class MultiImagePreview(NodeBase):
    label = "Multi Image Preview"
    category = "image"
    resizable = True
    params = {
        "images": {
            "label": "Images",
            "display": "input",
            "type": "image",
        },
        "image_list": {
            "display": "output",
            "type": "image",
            "hidden": True,
        },
        "preview": {
            "label": "Preview",
            "display": "ui_image",
            "type": "url",
            "dataSource": "image_list",
        },
        "size_index": {
            "label": "Size Index",
            "type": "int",
            "default": 0,
            "min": 0,
            "description": "Zero-based index of the image to get width/height from",
        },
        "image": {
            "label": "Image",
            "display": "output",
            "type": "image",
        },
        "width": {
            "label": "Width",
            "display": "output",
            "type": "int",
        },
        "height": {
            "label": "Height",
            "display": "output",
            "type": "int",
        },
    }

    def execute(self, **kwargs):
        images = kwargs.get("images")
        size_index = kwargs.get("size_index", 0)

        if images is None:
            return {"image_list": None, "image": None, "width": None, "height": None}

        image_list = flatten_images(images)

        if not image_list:
            return {"image_list": None, "image": None, "width": None, "height": None}

        # clamp size_index to valid range
        if size_index < 0:
            size_index = 0
        if size_index >= len(image_list):
            size_index = len(image_list) - 1

        selected_image = image_list[size_index]
        width = selected_image.width
        height = selected_image.height

        return {"image_list": image_list, "image": selected_image, "width": width, "height": height}


class ExtractImage(NodeBase):
    label = "Extract Image"
    category = "image"
    resizable = True
    params = {
        "images": {
            "label": "Images",
            "display": "input",
            "type": "image",
        },
        "index": {
            "label": "Index",
            "type": "int",
            "default": 0,
            "min": 0,
            "description": "Zero-based index of the image to extract",
        },
        "channels": {
            "label": "Channels",
            "type": "string",
            "default": "Keep Original",
            "options": ["Keep Original", "RGB (3 channels)", "RGBA (4 channels)"],
            "description": "Convert output image to specified channel mode",
        },
        "image": {
            "label": "Image",
            "display": "output",
            "type": "image",
        },
        "preview": {
            "label": "Preview",
            "display": "ui_image",
            "type": "url",
            "dataSource": "image",
        },
    }

    def execute(self, **kwargs):
        images = kwargs.get("images")
        index = kwargs.get("index", 0)
        channels = kwargs.get("channels", "Keep Original")

        if images is None:
            return {"image": None}

        # process nested lists
        flat_images = flatten_images(images)

        if not flat_images:
            self.notify(
                "No images found in input",
                variant="warning",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return {"image": None}

        # clamp index to valid range
        if index < 0:
            index = 0
        if index >= len(flat_images):
            self.notify(
                f"Index {index} is out of range. Max index is {len(flat_images) - 1}. Using last image.",
                variant="warning",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            index = len(flat_images) - 1

        image = flat_images[index]

        # convert to specified channel mode
        image = convert_channels(image, channels)

        return {"image": image}


class LoadImage(NodeBase):
    """
    Load an image from a file with optional channel conversion.
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
        "label": {
            "display": "ui_label",
            "value": "Load Image",
        },
        "file": {
            "label": False,
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {
                "fileTypes": ["image"],
                "multiple": True,
            },
        },
        "channels": {
            "label": "Channels",
            "type": "string",
            "default": "Keep Original",
            "options": ["Keep Original", "RGB (3 channels)", "RGBA (4 channels)"],
            "description": "Convert output image(s) to specified channel mode",
        },
        "size_index": {
            "label": "Size Index",
            "type": "int",
            "default": 0,
            "min": 0,
            "description": "Zero-based index of the image to get width/height from",
        },
        "scale_factor": {
            "label": "Scale Factor",
            "type": "string",
            "default": "1x",
            "options": ["0.2x", "0.4x", "0.5x", "0.6x", "0.8x", "1x", "2x", "3x", "4x", "6x", "8x", "10x"],
            "description": "Scale factor for width/height output (maintains aspect ratio)",
        },
        "width": {"label": "Width", "display": "output", "type": "int"},
        "height": {"label": "Height", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        file = kwargs.get("file")
        channels = kwargs.get("channels", "Keep Original")
        size_index = kwargs.get("size_index", 0)
        scale_factor_str = kwargs.get("scale_factor", "1x")
        images = []

        # parse scale factor
        scale_factor = float(scale_factor_str.replace("x", ""))

        file = file if isinstance(file, list) else [file]
        for f in file:
            if f is None or f == "":
                continue
            try:
                image = None

                if isinstance(f, str) and f.startswith(("http://", "https://")):
                    image = load_image(f)
                else:
                    if not Path(f).is_absolute():
                        f = Path(CONFIG.paths["work_dir"]) / f
                    if not Path(f).exists():
                        continue
                    image = Image.open(f)

                # convert to specified channel mode
                image = convert_channels(image, channels)
                images.append(image)
            except Exception as e:
                self.notify(
                    f"Error loading image {f}: {e}",
                    variant="error",
                    persist=False,
                    autoHideDuration=MESSAGE_DURATION,
                )
                continue

        if len(images) == 0:
            return {"image": None, "width": None, "height": None}

        # clamp size_index to valid range
        if size_index < 0:
            size_index = 0
        if size_index >= len(images):
            size_index = len(images) - 1

        selected_image = images[size_index]
        width = int(selected_image.width * scale_factor)
        height = int(selected_image.height * scale_factor)

        if len(images) == 1:
            images = images[0]

        return {"image": images, "width": width, "height": height}
