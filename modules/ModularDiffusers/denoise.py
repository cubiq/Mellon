import logging
from typing import Any, List, Tuple

import torch
from diffusers import ComponentsManager
from diffusers.modular_pipelines import BlockState, InputParam, PipelineBlock, SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS

from mellon.NodeBase import NodeBase

from . import components


logger = logging.getLogger("mellon")


class PreviewBlock(PipelineBlock):
    model_name = "stable-diffusion-xl"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("callback", default=None),
        ]

    def __call__(self, components: ComponentsManager, block_state: BlockState, i: int, t: int):
        block_state.callback(block_state.latents, i)

        return components, block_state


t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(ALL_BLOCKS["text2img"])
text_blocks = t2i_blocks.sub_blocks.pop("text_encoder")
decoder_blocks = t2i_blocks.sub_blocks.pop("decode")

preview_block = PreviewBlock()
t2i_blocks.sub_blocks["denoise"].sub_blocks.insert("preview_block", preview_block, 3)


class Denoise(NodeBase):
    label = "Denoise"
    category = "sampler"
    resizable = True
    params = {
        "unet": {"label": "Unet", "display": "input", "type": "diffusers_auto_model"},
        "scheduler": {"label": "Scheduler", "display": "input", "type": "scheduler"},
        "embeddings": {"label": "Text Embeddings", "display": "input", "type": "embeddings"},
        "latents": {"label": "Latents", "type": "latents", "display": "output"},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 64, "step": 8},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 64, "step": 8},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "steps": {"label": "Steps", "type": "int", "display": "slider", "default": 30, "min": 1, "max": 100},
        "cfg": {
            "label": "Guidance",
            "type": "float",
            "display": "slider",
            "default": 5,
            "min": 0.0,
            "max": 50.0,
            "step": 0.5,
        },
        "latents_preview": {"label": "Latents Preview", "display": "output", "type": "latent"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._denoise_node = t2i_blocks.init_pipeline(components_manager=components)

    def execute(self, unet, scheduler, width, height, embeddings, seed, steps, cfg):
        logger.debug(f" Denoise ({self.node_id}) received parameters:")
        logger.debug(f" - unet: {unet}")
        logger.debug(f" - scheduler: {scheduler}")
        logger.debug(f" - embeddings: {embeddings}")
        logger.debug(f" - width: {width}")
        logger.debug(f" - height: {height}")
        logger.debug(f" - seed: {seed}")
        logger.debug(f" - steps: {steps}")
        logger.debug(f" - cfg: {cfg}")

        unet_component = components.get_one(unet["model_id"])
        scheduler_component = components.get_one(scheduler["model_id"])
        self._denoise_node.update_components(unet=unet_component, scheduler=scheduler_component)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        guider_spec = self._denoise_node.get_component_spec("guider")
        guider_spec.config["guidance_scale"] = cfg
        self._denoise_node.update_components(guider=guider_spec)

        def preview_callback(latents, step: int):
            self.trigger_output("latents_preview", latents)
            progress = int((step + 1) / steps * 100)
            self.progress(progress)

        latents = self._denoise_node(
            **embeddings,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
            output="latents",
            callback=preview_callback,
        )

        return {"latents": latents, "latents_preview": latents}
