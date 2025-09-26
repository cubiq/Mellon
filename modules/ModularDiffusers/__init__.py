from diffusers import ComponentsManager


components = ComponentsManager()

MODULE_PARSE = ["adapters", "controlnet", "denoise", "embeddings", "guiders", "latents", "loaders", "schedulers", "custom"]

SDXL_BLOCKS = [
    "down_blocks.1.attentions.0",
    "down_blocks.1.attentions.1",
    "down_blocks.2.attentions.0",
    "down_blocks.2.attentions.1",
    "mid_block.attentions.0",
    "up_blocks.0.attentions.0",
    "up_blocks.0.attentions.1",
    "up_blocks.0.attentions.2",
    "up_blocks.1.attentions.0",
    "up_blocks.1.attentions.1",
    "up_blocks.1.attentions.2  ",
]
