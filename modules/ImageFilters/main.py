from mellon.NodeBase import NodeBase
from utils.torch_utils import ImageToTensor, TensorToImage

class Canny(NodeBase):
    def execute(self, **kwargs):
        from kornia.filters import canny

        image = kwargs.get("image")
        low_threshold = kwargs.get("low_threshold", 0.1)
        high_threshold = kwargs.get("high_threshold", 0.2)
        low_threshold = min(low_threshold, high_threshold)
        high_threshold = max(low_threshold, high_threshold)
        device = kwargs.get("device")

        image = ImageToTensor(image)
        image = image if isinstance(image, list) else [image]

        output = []
        for i in image:
            if i.ndim == 3:
                i = i.unsqueeze(0)

            _, edges = canny(i.to(device), low_threshold, high_threshold)
            output.append(edges.to('cpu'))
        del image
        
        output = TensorToImage(output)

        return { "output": output }
    

class UnsharpMask(NodeBase):
    def execute(self, **kwargs):
        from kornia.filters import unsharp_mask

        image = kwargs.get("image")
        radius = kwargs.get("radius", 3)
        # radius must be an odd number
        if radius % 2 == 0:
            radius += 1
        radius = (radius, radius)

        amount = kwargs.get("amount", 0.3)
        amount = radius[0] / 3 * amount
        amount = (amount, amount)

        device = kwargs.get("device")
        
        image = ImageToTensor(image)
        image = image if isinstance(image, list) else [image]

        output = []
        for i in image:
            if i.ndim == 3:
                i = i.unsqueeze(0)
            
            sharp = unsharp_mask(i.to(device), radius, amount)
            output.append(sharp.to('cpu'))
        del image
        
        output = TensorToImage(output)

        return { "output": output }

class GuidedBlur(NodeBase):
    def execute(self, **kwargs):
        from kornia.filters import guided_blur
        from utils.image import fit

        image = kwargs.get("image")
        radius = kwargs.get("radius", 5)
        eps = kwargs.get("eps", 0.1)
        subsample = kwargs.get("subsample", 1)
        device = kwargs.get("device")

        # radius must be an odd number
        if radius % 2 == 0:
            radius += 1
        radius = (radius, radius)

        image = image if isinstance(image, list) else [image]
        if subsample > 1:
            # ensure the image is divisible by the subsample factor
            image = [fit(i, i.width // subsample * subsample, i.height // subsample * subsample) for i in image]

        image = ImageToTensor(image)

        output = []
        for i in image:
            if i.ndim == 3:
                i = i.unsqueeze(0)
            
            blur = guided_blur(i.to(device), i.to(device), kernel_size=radius, eps=eps, subsample=subsample)
            output.append(blur.to('cpu'))
        del image
        
        output = TensorToImage(output)

        return { "output": output }