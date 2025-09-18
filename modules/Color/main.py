
from mellon.NodeBase import NodeBase
from PIL import Image, ImageOps

class Invert(NodeBase):
    """
    Invert the colors of an image
    """

    label = "Invert Colors"
    category = "color"
    params = {
        "image": {
            "label": "Image",
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
        image = kwargs.get("image", None)

        if image is None:
            return {"output": None}

        image = [image] if not isinstance(image, list) else image

        # Invert the colors
        inverted = []
        for img in image:
            # invert the colors ignoring the alpha channel
            alpha = img.split()[-1] if img.mode == 'RGBA' else None
            img = img.convert("RGB")
            #img = Image.eval(img, lambda x: 255 - x) # cool kids' way
            img = ImageOps.invert(img)
            if alpha:
                img.putalpha(alpha)
            inverted.append(img)

        return {"output": inverted}
