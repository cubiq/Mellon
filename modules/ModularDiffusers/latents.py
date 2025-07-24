import logging

import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS
from PIL import Image

from mellon.NodeBase import NodeBase

from . import components


logger = logging.getLogger("mellon")

t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(ALL_BLOCKS["text2img"])
decoder_blocks = t2i_blocks.sub_blocks.pop("decode")


class LatentsPreview(NodeBase):
    label = "Latents Preview"
    category = "image"
    resizable = True
    params = {
        "latents": {"label": "Latents", "display": "input", "type": "latent"},
        "image": {"label": "Image", "display": "output", "type": "image", "hidden": True},
        "preview": {"display": "ui_image", "dataSource": "image"},
    }

    def execute(self, latents):
        latent_rgb_factors = [
            [0.3920, 0.4054, 0.4549],
            [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076],
        ]

        image = None

        if latents is not None:
            latent_rgb_factors = torch.tensor(latent_rgb_factors, dtype=latents.dtype).to(device=latents.device)
            latent_image = latents.squeeze(0).permute(1, 2, 0) @ latent_rgb_factors
            latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
            denoised_image = latents_ubyte.byte().cpu().numpy()
            image = Image.fromarray(denoised_image)
            image = image.resize((image.width * 2, image.height * 2), resample=Image.Resampling.BICUBIC)

        return {"image": image}


class DecodeLatents(NodeBase):
    label = "Decode Latents"
    category = "sampler"
    resizable = True
    params = {
        "vae": {"label": "VAE", "display": "input", "type": "diffusers_auto_model"},
        "latents": {"label": "Latents", "type": "latents", "display": "input"},
        "images": {"label": "Images", "type": "image", "display": "output"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._decoder_node = decoder_blocks.init_pipeline(components_manager=components)

    def execute(self, vae, latents):
        logger.debug(f" DecodeLatents ({self.node_id}) received parameters:")
        logger.debug(f" - vae: {vae}")
        logger.debug(f" - latents: {latents.shape}")

        vae_component = components.get_one(vae["model_id"])
        self._decoder_node.update_components(vae=vae_component)
        image = self._decoder_node(latents=latents, output="images")[0]

        return {"images": image}
