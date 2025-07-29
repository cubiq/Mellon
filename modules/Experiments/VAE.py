from mellon.NodeBase import NodeBase
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST
from utils.torch_utils import TensorToImage
import torch

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // int(vae_scale_factor * 2))
    width = 2 * (int(width) // int(vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

class VAEDecode(NodeBase):
    label = "VAE Decode"
    category = "sampler"
    params = {
        "images": { "label": "Images", "display": "output", "type": "image" },
        "pipeline": { "label": "VAE", "display": "input", "type": "pipeline" },
        "latents": { "label": "Latents", "display": "input", "type": "latent" },
        "tiling": { "label": "Enable Tiling", "type": "boolean", "default": False },
        "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
    }

    def execute(self, pipeline, **kwargs):
        latents = kwargs['latents']
        vae = pipeline.vae if hasattr(pipeline, 'vae') else pipeline
        device = kwargs.get('device', DEFAULT_DEVICE)
        tiling = kwargs.get('tiling', False)

        if tiling:
            vae.enable_tiling()
        else:
            vae.disable_tiling()

        self.mm_load(vae, device)
        images = self.mm_exec(lambda: self.decode(vae, latents), device, exclude=[vae])
        
        return { "images": images }

    def decode(self, model, latents, size=None):
        if hasattr(model, 'post_quant_conv') and hasattr(model.post_quant_conv, 'parameters'):
            latents = latents.to(dtype=next(iter(model.post_quant_conv.parameters())).dtype)
        else:
            latents = latents.to(dtype=model.dtype)
        
        if size is not None:
            latents = unpack_latents(latents, size[0], size[1], 2 ** (len(model.config.block_out_channels) - 1))

        #latents = 1 / model.config['scaling_factor'] * latents
        latents = (latents / model.config.scaling_factor) + model.config.shift_factor
        images = model.decode(latents.to(model.device), return_dict=False)[0]
        latents = latents.to('cpu')
        del latents, model

        images = images / 2 + 0.5
        images = TensorToImage(images.to('cpu').detach().clone())
        return images
    
    