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
        for img in image:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.to(device)
            blur = guided_blur(img, img, kernel_size=radius, eps=eps, subsample=subsample)
            output.append(blur.to('cpu'))
        del image
        
        output = TensorToImage(output)

        return { "output": output }
    
class AdaptiveSharpening(NodeBase):
    def execute(self, **kwargs):
        image = kwargs.get("image")
        sharpness = kwargs.get("sharpness", 0.8)
        device = kwargs.get("device")

        image = ImageToTensor(image)
        image = image if isinstance(image, list) else [image]

        output = []
        for img in image:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            
            # Move image to the correct device before processing
            img = img.to(device)

            # Call the static sharpen method
            sharp_img = AdaptiveSharpening.sharpen(img, sharpness)
            output.append(sharp_img.to('cpu'))
        
        del image

        output = TensorToImage(output)

        return { "output": output }
    
    @staticmethod
    def sharpen(image, sharpness):
        import torch
        import torch.nn.functional as F

        epsilon = 1e-5
        
        # Use Unfold to extract 3x3 patches
        unfold = torch.nn.Unfold(kernel_size=3, padding=1, stride=1)
        patches = unfold(image).view(image.shape[0], image.shape[1], 9, image.shape[2], image.shape[3])

        # Center pixel is at index 4
        e = patches[:, :, 4]

        # Indices for neighbors
        # 0 1 2
        # 3 4 5
        # 6 7 8
        cross_indices = [1, 3, 5, 7]
        diag_indices = [0, 2, 6, 8]
        
        # Extract cross and diagonal neighbors using indexing
        cross_neighbors = patches[:, :, cross_indices]
        diag_neighbors = patches[:, :, diag_indices]

        # Computing contrast
        # Note: The original code included the center pixel 'e' in the 'cross' min/max.
        # To replicate that, we can concatenate it.
        cross_and_center = torch.cat((cross_neighbors, e.unsqueeze(2)), dim=2)
        mn, _ = torch.min(cross_and_center, dim=2)
        mx, _ = torch.max(cross_and_center, dim=2)

        mn2, _ = torch.min(diag_neighbors, dim=2)
        mx2, _ = torch.max(diag_neighbors, dim=2)
        
        mx = mx + mx2
        mn = mn + mn2

        # Computing local weight
        inv_mx = torch.reciprocal(mx + epsilon)
        amp = inv_mx * torch.minimum(mn, (2 - mx))

        # scaling
        amp = torch.sqrt(torch.clamp(amp, min=0)) # Add clamp for stability
        w = -amp * (sharpness * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        # Sum cross neighbors and apply weights
        sum_cross_neighbors = torch.sum(cross_neighbors, dim=2)
        output = (sum_cross_neighbors * w + e) * div
        output = output.clamp(0, 1)

        return output

class GaussianBlur(NodeBase):
    def execute(self, **kwargs):
        from kornia.filters import gaussian_blur2d
        image = kwargs.get("image")
        amount = kwargs.get("amount", 1)
        device = kwargs.get("device")

        sigma_value = max(0.1, min(100, amount))
        sigma = (sigma_value, sigma_value)

        # Calculate kernel_size based on sigma
        ksize = int(2 * round(3 * sigma_value) + 1)
        kernel_size = (ksize, ksize)

        device = kwargs.get("device")
        image = image if isinstance(image, list) else [image]
        image = ImageToTensor(image)

        # # kernel_size must be an odd number
        # if radius % 2 == 0:
        #     radius += 1
        # radius = (radius, radius)
        # sigma = (sigma, sigma)

        image = image if isinstance(image, list) else [image]
        image = ImageToTensor(image)

        output = []
        for img in image:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.to(device)
            blur = gaussian_blur2d(img, kernel_size=kernel_size, sigma=sigma)
            output.append(blur.to('cpu'))
        del image

        output = TensorToImage(output)

        return { "output": output }