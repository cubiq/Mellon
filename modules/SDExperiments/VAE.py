from mellon.NodeBase import NodeBase
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST
from PIL import Image
import torch
from utils.torch_utils import TensorToImage
#from utils.memory_menager import memory_flush

class VAEDecode(NodeBase):
    label = "VAE Decode"
    category = "decoder"
    params = {
        "vae": { "label": "VAE", "display": "input", "type": "pipeline" },
        "latents": { "label": "Latents", "display": "input", "type": "latent" },
        "images": { "label": "Images", "display": "output", "type": "image" },
        "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
    }

    def execute(self, **kwargs):
        latents = kwargs['latents']
        vae = kwargs['vae'].vae if hasattr(kwargs['vae'], 'vae') else kwargs['vae']
        device = kwargs.get('device', DEFAULT_DEVICE)

        self.mm_load(vae, device=device)
        images = self.mm_exec(self.decode, device, exclude=[vae], args=[vae, latents])

        return { "images": images }

    def decode(self, model, latents):
        dtype = model.dtype
        
        if dtype == torch.float16 and model.config.force_upcast:
            self.upcast_vae(model)

        if hasattr(model, 'post_quant_conv') and hasattr(model.post_quant_conv, 'parameters'):
            latents = latents.to(dtype=next(iter(model.post_quant_conv.parameters())).dtype)
        else:
            latents = latents.to(dtype=model.dtype)

        latents = 1 / model.config['scaling_factor'] * latents
        images = model.decode(latents.to(model.device), return_dict=False)[0]
        del latents, model
        images = images / 2 + 0.5
        images = TensorToImage(images.to('cpu'))
        return images
    
    def upcast_vae(self, model):
        from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor

        dtype = model.dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            new_dtype = torch.bfloat16
        else:
            new_dtype = torch.float32

        model.to(dtype=new_dtype)
        use_torch_2_0_or_xformers = isinstance(
            model.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            model.post_quant_conv.to(dtype)
            model.decoder.conv_in.to(dtype)
            model.decoder.mid_block.to(dtype)    
    