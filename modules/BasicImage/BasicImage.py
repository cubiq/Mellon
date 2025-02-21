from PIL import Image
import torch
from utils.torch_utils import toTensor, toPIL
from mellon.NodeBase import NodeBase
from modules.VAE.VAE import VAEDecode

class Preview(VAEDecode):
    def execute(self, images, vae, device):
        if isinstance(images, torch.Tensor):
            if not vae:
                raise ValueError("VAE is required to decode latents")

            if hasattr(vae, 'vae'):
                vae = vae.vae

            self.mm_load(vae, device)
            images = self.mm_inference(
                lambda: self.vae_decode(vae, images),
                device,
                exclude=vae,
            )

        if not isinstance(images, list):
            images = [images]

        return {
            'images_out': images,
            'width': images[0].width,
            'height': images[0].height
        }


class LoadImage(NodeBase):
    def execute(self, path):
        return { 'images': Image.open(path) }


class SaveImage(NodeBase):
    def execute(self, images: list):
        # save all the images in the list
        for i, image in enumerate(images):
            image.save(f"image_{i}.webp")

        return

class Resize(NodeBase):
    def execute(self, images, width, height, method, resample):
        if width == 0 and height == 0:
            return { 'images_out': images, 'width': ow, 'height': oh }

        resample = resample.upper()
        if method == 'stretch':
            images = images.resize((max(width, 1), max(height, 1)), resample=Image.Resampling[resample])
        elif method == 'fit':
            from PIL.ImageOps import fit
            images = fit(images, (max(width, 1), max(height, 1)), method=Image.Resampling[resample])
        elif method == 'pad':
            from PIL.ImageOps import pad
            images = pad(images, (max(width, 1), max(height, 1)), Image.Resampling[resample])
        elif method == 'keep aspect ratio':
            ow, oh = images.size
            if width == 0:
                scale = height / oh
                width = int(ow * scale)
            elif height == 0:
                scale = width / ow
                height = int(oh * scale)
            else:
                scale = min(width / ow, height / oh)
                new_width = int(ow * scale)
                new_height = int(oh * scale)
                # prevent rounding errors
                if height / oh < width / ow:
                    new_height = height
                elif width / ow < height / oh:
                    new_width = width

            images = images.resize((max(new_width, 1), max(new_height, 1)), resample=Image.Resampling[resample])

        return { 'images_out': images,
                 'width': images.width,
                 'height': images.height }

class ScaleBy(NodeBase):
    def execute(self, images, factor, resample):
        images = images.resize((max(int(images.width * factor), 1), max(int(images.height * factor), 1)), resample=Image.Resampling[resample.upper()])
        return { 'images_out': images,
                 'width': images.width,
                 'height': images.height }

class ResizeToDivisible(NodeBase):
    def execute(self, images, divisible_by):
        from PIL.ImageOps import fit
        divisible_by = int(max(1, divisible_by))
        width, height = images.size
        width = width // divisible_by * divisible_by
        height = height // divisible_by * divisible_by
        images = fit(images, (width, height), method=Image.Resampling.LANCZOS)
        return { 'images_out': images,
                 'width': width,
                 'height': height }

# NOT implemented
class BlendImages(NodeBase):
    def execute(self, source: Image.Image, target: Image.Image, amount: float):
        source = toTensor(source)
        target = toTensor(target)
        blend = source * amount + target * (1 - amount)
        blend = toPIL(blend)

        return { 'blend': blend }
