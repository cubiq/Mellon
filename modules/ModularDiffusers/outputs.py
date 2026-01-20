from pathlib import Path

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
        "count": {
            "label": "Image Count",
            "display": "output",
            "type": "int",
        },
    }

    def execute(self, **kwargs):
        images = kwargs.get("images")

        if images is None:
            return {"image_list": None, "count": 0}

        image_list = flatten_images(images)

        if not image_list:
            return {"image_list": None, "count": 0}

        return {"image_list": image_list, "count": len(image_list)}


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
        "width": {"display": "output", "type": "int"},
        "height": {"display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        file = kwargs.get("file")
        channels = kwargs.get("channels", "Keep Original")
        images = []
        widths = []
        heights = []

        file = file if isinstance(file, list) else [file]
        for f in file:
            if f is None or f == "":
                continue
            if not Path(f).is_absolute():
                f = Path(CONFIG.paths["work_dir"]) / f
            if not Path(f).exists():
                continue
            try:
                image = Image.open(f)

                # convert to specified channel mode
                image = convert_channels(image, channels)
                images.append(image)
                widths.append(image.width)
                heights.append(image.height)
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

        if len(images) == 1:
            images = images[0]
            widths = widths[0]
            heights = heights[0]

        return {"image": images, "width": widths, "height": heights}
