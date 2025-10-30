import logging
from typing import Any, List, Tuple
import importlib
from .modular_utils import pipeline_class_to_mellon_node_config, get_model_type_signal_data

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

# SIGNAL_DATA = get_model_type_signal_data()

class Denoise(NodeBase):
    label = "Denoise"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    node_type = "denoise"
    params = {
        "model_type": {
            "label": "Model Type", 
            "type": "string", 
            "default": "", 
            "hidden": True  # Hidden field to receive signal data
        },
        "unet": {
            "label": "Denoise Model",
            "display": "input",
            "type": "diffusers_auto_model",
            "onSignal": [
                {
                    "action": "value",
                    "target": "model_type",
                    # "data": SIGNAL_DATA, # YiYi Notes: not working
                    "data": {
                        "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
                        "QwenImageModularPipeline": "QwenImageModularPipeline",
                        "QwenImageEditModularPipeline": "QwenImageEditModularPipeline",
                        "QwenImageEditPlusModularPipeline": "QwenImageEditPlusModularPipeline",
                        "FluxModularPipeline": "FluxModularPipeline",
                        "FluxKontextModularPipeline": "FluxKontextModularPipeline",
                    },
                },
                {"action": "exec", "data": "update_node"},
                {"action": "signal", "target": "guider"},
                {"action": "signal", "target": "controlnet"},
            ]
        },
    }

    def update_node(self, values, ref):

        node_params  = {
            "model_type": {
                "label": "Model Type", 
                "type": "string", 
                "default": "", 
                "hidden": True  # Hidden field to receive signal data
            },
            "unet": {
                "label": "Denoise Model",
                "display": "input",
                "type": "diffusers_auto_model",
                "onSignal": [
                    {
                        "action": "value",
                        "target": "model_type",
                        # "data": SIGNAL_DATA, # YiYi Notes: not working
                        "data": {
                            "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
                            "QwenImageModularPipeline": "QwenImageModularPipeline",
                            "QwenImageEditModularPipeline": "QwenImageEditModularPipeline",
                            "QwenImageEditPlusModularPipeline": "QwenImageEditPlusModularPipeline",
                            "FluxModularPipeline": "FluxModularPipeline",
                            "FluxKontextModularPipeline": "FluxKontextModularPipeline",
                        },
                    },
                    {"action": "exec", "data": "update_node"},
                    {"action": "signal", "target": "guider"},
                    {"action": "signal", "target": "controlnet"},
                ]
            },
        }
        model_type = values.get("model_type", "")

        if model_type == "" or self._model_type == model_type:
            return None

        self._model_type = model_type

        diffusers_module = importlib.import_module("diffusers")
        self._pipeline_class = getattr(diffusers_module, model_type)

        _, denoise_mellon_config = pipeline_class_to_mellon_node_config(self._pipeline_class, self.node_type)
        # not support this node type
        if denoise_mellon_config is None:
            self.send_node_definition(node_params)
            return

        # required params for controlnet
        node_params.update(**denoise_mellon_config.to_mellon_dict()["params"])
        self.send_node_definition(node_params)

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

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
        skip_image_size=False,
        **kwargs,
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

        # qwen image edit models work better without setting the image size
        if skip_image_size:
            width = height = None
        else:
            width = int(width) if width is not None else None
            height = int(height) if height is not None else None

        denoise_kwargs = {
            **embeddings,
            "num_inference_steps": int(num_inference_steps),
            "generator": generator,
            "width": width,
            "height": height,
        }

        if guider is None:
            try:
                guider_spec = self._denoise_node.get_component_spec("guider")
            except Exception:
                logger.debug("Guider component not found, this should mean that the model does not require one.")
            else:
                if guider_spec is not None:
                    guider_spec.config["guidance_scale"] = guidance_scale
                    self._denoise_node.update_components(guider=guider_spec)
        else:
            self._denoise_node.update_components(guider=guider)

        if image_latents is not None:
            denoise_kwargs["image_latents"] = image_latents
            denoise_kwargs["strength"] = float(strength)

        if controlnet is not None:
            controlnet_inputs = dict(controlnet["controlnet_inputs"])
            controlnet_scale = float(controlnet_inputs.get("controlnet_conditioning_scale", 1.0))
            control_guidance_start = float(controlnet_inputs.get("control_guidance_start", 0.0))
            control_guidance_end = float(controlnet_inputs.get("control_guidance_end", 1.0))
            controlnet_width = int(controlnet_inputs.get("width", width))
            controlnet_height = int(controlnet_inputs.get("height", height))
            denoise_kwargs.update(
                {
                    **controlnet_inputs,
                    "controlnet_conditioning_scale": controlnet_scale,
                    "control_guidance_start": control_guidance_start,
                    "control_guidance_end": control_guidance_end,
                    "width": controlnet_width,
                    "height": controlnet_height,
                }
            )

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

        return {
            "latents": latents_dict,
            "latents_preview": latents_dict["latents"],
            "doc": self._denoise_node.blocks.doc,
        }
