from mellon.NodeBase import NodeBase
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST
from utils.torch_utils import TensorToImage
import torch

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

    def decode(self, model, latents):   
        if hasattr(model, 'post_quant_conv') and hasattr(model.post_quant_conv, 'parameters'):
            latents = latents.to(dtype=next(iter(model.post_quant_conv.parameters())).dtype)
        else:
            latents = latents.to(dtype=model.dtype)

        latents = 1 / model.config['scaling_factor'] * latents
        images = model.decode(latents.to(model.device), return_dict=False)[0]
        latents = latents.to('cpu')
        del latents, model

        images = images / 2 + 0.5
        images = TensorToImage(images.to('cpu').detach().clone())
        return images
    
    