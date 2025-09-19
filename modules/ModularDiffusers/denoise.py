import logging
from typing import Any, List, Tuple

import torch
from diffusers import ComponentsManager
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from diffusers.modular_pipelines import (
    BlockState,
    InputParam,
    LoopSequentialPipelineBlocks,
    ModularPipeline,
    ModularPipelineBlocks,
)

from mellon.NodeBase import NodeBase

from . import components


logger = logging.getLogger("mellon")


class PreviewBlock(ModularPipelineBlocks):
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("callback", default=None),
        ]

    def __call__(self, components: ComponentsManager, block_state: BlockState, i: int, t: int):
        block_state.callback(block_state.latents, i, components.scheduler.order)

        return components, block_state


def insert_preview_block(pipeline):
    """Insert preview_block into all LoopSequentialPipelineBlocks with 'denoise' in name."""

    # new block so we can preview the generation
    preview_block = PreviewBlock()

    def insert_preview_block_recursive(blocks, blocks_name, preview_block):
        if hasattr(blocks, "sub_blocks"):
            if isinstance(blocks, LoopSequentialPipelineBlocks) and "denoise" in blocks_name.lower():
                blocks.sub_blocks.insert("preview_block", preview_block, len(blocks.sub_blocks))
            else:
                for sub_block_name, sub_block in blocks.sub_blocks.items():
                    insert_preview_block_recursive(sub_block, sub_block_name, preview_block)

    insert_preview_block_recursive(pipeline.blocks.sub_blocks["denoise"], "", preview_block)


class Denoise(NodeBase):
    label = "Denoise"
    category = "sampler"
    resizable = True
    params = {
        "unet": {
            "label": "Denoise Model",
            "display": "input",
            "type": "diffusers_auto_model",
            "onSignal": [
                {"action": "signal", "target": "guider"},
                {"action": "signal", "target": "controlnet"},
                {
                    "StableDiffusionXLModularPipeline": ["width", "height", "ip_adapter", "controlnet"],
                    "QwenImageModularPipeline": ["width", "height", "controlnet"],
                    "": [],
                },
            ],
        },
        "scheduler": {"label": "Scheduler", "display": "input", "type": "diffusers_auto_model"},
        "embeddings": {"label": "Text Embeddings", "display": "input", "type": "embeddings"},
        "latents": {"label": "Latents", "type": "latents", "display": "output"},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 64, "step": 8},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 64, "step": 8},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "num_inference_steps": {
            "label": "Steps",
            "type": "int",
            "display": "slider",
            "default": 25,
            "min": 1,
            "max": 100,
        },
        "guidance_scale": {
            "label": "Guidance Scale",
            "type": "float",
            "display": "slider",
            "default": 5,
            "min": 1.0,
            "max": 30.0,
            "step": 0.1,
        },
        "latents_preview": {"label": "Latents Preview", "display": "output", "type": "latent"},
        "guider": {
            "label": "Guider",
            "display": "input",
            "type": "custom_guider",
            "onChange": {False: ["guidance_scale"], True: []},
        },
        "image_latents": {
            "label": "Image Latents",
            "type": "latents",
            "display": "input",
            "onChange": {False: ["height", "width"], True: ["strength"]},
        },
        "strength": {
            "label": "Strength",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "controlnet": {
            "label": "Controlnet",
            "type": "custom_controlnet",
            "display": "input",
        },
        "ip_adapter": {
            "label": "IP Adapter",
            "type": "custom_ip_adapter",
            "display": "input",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._denoise_node = None

    def execute(
        self,
        unet,
        scheduler,
        embeddings,
        seed,
        num_inference_steps,
        guidance_scale,
        guider=None,
        image_latents=None,
        strength=0.5,
        controlnet=None,
        ip_adapter=None,
        width=None,
        height=None,
    ):
        logger.debug(f" Denoise ({self.node_id}) received parameters:")
        logger.debug(f" - unet: {unet}")
        logger.debug(f" - scheduler: {scheduler}")
        logger.debug(f" - embeddings: {embeddings}")
        logger.debug(f" - width: {width}")
        logger.debug(f" - height: {height}")
        logger.debug(f" - seed: {seed}")
        logger.debug(f" - num_inference_steps: {num_inference_steps}")
        logger.debug(f" - guidance_scale: {guidance_scale}")

        device = unet["execution_device"]
        repo_id = unet["repo_id"]
        # derive auto blocks from repo_id
        auto_blocks = ModularPipeline.from_pretrained(repo_id).blocks
        denoise_blocks = auto_blocks.sub_blocks.pop("denoise")

        self._denoise_node = denoise_blocks.init_pipeline(repo_id, components_manager=components)

        insert_preview_block(self._denoise_node)

        def preview_callback(latents, step_index: int, scheduler_order: int):
            if (step_index + 1) % scheduler_order == 0:
                self.trigger_output("latents_preview", latents)
                progress = int((step_index + 1) / num_inference_steps * 100 / scheduler_order)
                self.progress(progress)

        unet_component_dict = components.get_components_by_ids(ids=[unet["model_id"]], return_dict_with_names=True)
        scheduler_component_dict = components.get_components_by_ids(
            ids=[scheduler["model_id"]], return_dict_with_names=True
        )
        self._denoise_node.update_components(**scheduler_component_dict, **unet_component_dict)

        generator = torch.Generator(device=device).manual_seed(seed)

        denoise_kwargs = {
            **embeddings,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "width": width,
            "height": height,
        }

        if guider is None:
            guider_spec = self._denoise_node.get_component_spec("guider")
            guider_spec.config["guidance_scale"] = guidance_scale
            self._denoise_node.update_components(guider=guider_spec)
        else:
            self._denoise_node.update_components(guider=guider)

        if image_latents is not None:
            denoise_kwargs["image_latents"] = image_latents
            denoise_kwargs["strength"] = strength

        if controlnet is not None:
            denoise_kwargs.update(**controlnet["controlnet_inputs"])

            model_ids = controlnet["controlnet_model"]["model_id"]
            if isinstance(model_ids, list):
                # TODO: this is not working yet
                controlnet_components = [components.get_one(model_id) for model_id in model_ids]
                controlnet_components = MultiControlNetModel(controlnet_components)
            else:
                controlnet_components = components.get_one(model_ids)

            self._denoise_node.update_components(controlnet=controlnet_components)

        if ip_adapter is not None:
            self._denoise_node.load_ip_adapter(
                ip_adapter["model_id"], ip_adapter["subfolder"], ip_adapter["weight_name"]
            )
            self._denoise_node.set_ip_adapter_scale(ip_adapter["scale"])

            denoise_kwargs.update(ip_adapter_image=ip_adapter["image"])  # TODO: use embeddings instead

            image_encoder_input = ip_adapter["image_encoder"]
            image_encoder_component = components.get_one(image_encoder_input["model_id"])
            self._denoise_node.update_components(image_encoder=image_encoder_component)

        state = self._denoise_node(**denoise_kwargs, callback=preview_callback)
        latents_dict = {
            "latents": state.get("latents"),
            "height": state.get("height"),
            "width": state.get("width"),
        }

        return {"latents": latents_dict, "latents_preview": latents_dict["latents"]}
