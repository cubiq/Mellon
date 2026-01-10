import logging

import torch
from diffusers import ComponentSpec, ModularPipeline
from diffusers.modular_pipelines.mellon_node_utils import MellonPipelineConfig

from mellon.NodeBase import NodeBase
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

from . import MESSAGE_DURATION, components
from .modular_utils import (
    DUMMY_CUSTOM_PIPELINE_CONFIG,
    MODULAR_REGISTRY,
    DummyCustomPipeline,
    get_all_model_types,
    get_model_type_metadata,
)


logger = logging.getLogger("mellon")
logger.setLevel(logging.DEBUG)


def node_get_component_info(node_id=None, manager=None, name=None):
    comp_ids = manager._lookup_ids(name=name, collection=node_id)
    if len(comp_ids) != 1:
        raise ValueError(f"Expected 1 component for {name} for node {node_id}, got {len(comp_ids)}")
    return manager.get_model_info(list(comp_ids)[0])


def update_lora_adapters(lora_node, lora_list):
    """
    Update LoRA adapters based on the provided list of LoRAs.

    Args:
        lora_node: ModularPipeline node containing LoRA functionality
        lora_list: List of dictionaries or single dictionary containing LoRA configurations with:
                  {'lora_path': str, 'weight_name': str, 'adapter_name': str, 'scale': float}
    """
    # Convert single lora to list if needed
    if not isinstance(lora_list, list):
        lora_list = [lora_list]

    # Get currently loaded adapters
    loaded_adapters = list(set().union(*lora_node.get_list_adapters().values()))

    # Determine which adapters to set and remove
    to_set = [lora["adapter_name"] for lora in lora_list]
    to_remove = [adapter for adapter in loaded_adapters if adapter not in to_set]

    # Remove unused adapters first
    for adapter_name in to_remove:
        lora_node.delete_adapters(adapter_name)

    # Load new LoRAs and set their scales
    scales = {}
    for lora in lora_list:
        adapter_name = lora["adapter_name"]
        if adapter_name not in loaded_adapters:
            lora_node.load_lora_weights(
                lora["lora_path"],
                weight_name=lora["weight_name"],
                adapter_name=adapter_name,
            )
        scales[adapter_name] = lora["scale"]

    # Set adapter scales
    if scales:
        lora_node.set_adapters(list(scales.keys()), list(scales.values()))


class AutoModelLoader(NodeBase):
    label = "Load Model"
    category = "loader"
    resizable = True
    skipParamsCheck = True
    params = {
        "model_type": {
            "label": "Model Type",
            "type": "string",
            "options": {
                "": "",
                "denoise": "Denoise Model",
                "vae": "VAE",
                "controlnet": "ControlNet",
                "image_encoder": "Image Encoder",
            },
            "onChange": [
                "set_filters",
                {"action": "signal", "target": "model"},
            ],
        },
        "model_id": {
            "label": "Model ID",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": [""]},
                    "local": {"className": [""]},
                },
            },
        },
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "value": "float16",
            "postProcess": str_to_dtype,
        },
        "subfolder": {"label": "Subfolder", "type": "string", "value": ""},
        "variant": {"type": "string", "value": "", "options": ["", "fp16", "bf16"]},
        "model": {"label": "Model", "display": "output", "type": "diffusers_auto_model"},
    }

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        super().__del__()

    def set_filters(self, values, ref):
        model_type = values.get("model_type", "")

        filters = []

        if model_type == "denoise":
            filters = ["UNet2DConditionModel", "QwenImageTransformer2DModel", "FluxTransformer2DModel"]
            self.set_field_params("subfolder", {"value": "unet"})
        elif model_type == "vae":
            filters = ["AutoencoderKL", "AutoencoderKLQwenImage"]
            self.set_field_params("subfolder", {"value": "vae"})
        elif model_type == "controlnet":
            filters = ["ControlNetModel", "QwenImageControlNetModel", "FluxControlNetModel"]
            self.set_field_params("subfolder", {"value": ""})
        elif model_type == "image_encoder":
            filters = ["ClipVisionModel"]
            self.set_field_params("subfolder", {"value": "image_encoder"})

        default_values = {
            "": "",
            "denoise": "stabilityai/stable-diffusion-xl-base-1.0",
            "vae": "stabilityai/stable-diffusion-xl-base-1.0",
            "controlnet": "diffusers/controlnet-depth-sdxl-1.0",
            "image_encoder": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        }

        self.set_field_params(
            "model_id",
            {
                "default": {"source": "hub", "value": default_values[model_type]},
                "value": {"source": "hub", "value": default_values[model_type]},
                "fieldOptions": {
                    "filter": {
                        "hub": {"className": filters},
                    },
                },
            },
        )

    def execute(self, model_type, model_id, dtype, variant=None, subfolder=None):
        logger.debug(f"AutoModelLoader ({self.node_id}) received parameters:")
        logger.debug(f"  model_type: '{model_type}'")
        logger.debug(f"  model_id: '{model_id}'")
        logger.debug(f"  subfolder: '{subfolder}'")
        logger.debug(f"  variant: '{variant}'")
        logger.debug(f"  dtype: '{dtype}'")

        if isinstance(model_id, dict):
            real_model_id = model_id.get("value", model_id)
            _source = model_id.get("source", "hub")  # TODO: do something when is local?
        else:
            real_model_id = ""

        if real_model_id == "":
            self.notify(
                "Please provide a valid Repository ID.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        # Normalize parameters
        variant = None if variant == "" else variant
        subfolder = None if subfolder == "" else subfolder

        spec = ComponentSpec(name=model_type, repo=real_model_id, subfolder=subfolder, variant=variant)
        model = spec.load(torch_dtype=dtype)
        comp_id = components.add(model_type, model, collection=self.node_id)
        logger.debug(f" AutoModelLoader: comp_id added: {comp_id}")
        logger.debug(f" AutoModelLoader: component manager: {components}")

        model = components.get_model_info(comp_id)
        model["repo_id"] = real_model_id

        return {"model": model}


class ModelsLoader(NodeBase):
    label = "Load Models"
    category = "loader"
    resizable = True
    skipParamsCheck = True
    params = {
        "model_type": {
            "label": "Model Type",
            "type": "string",
            "options": {
                "": "",
            },
            "onChange": [
                "set_filters",
                {"action": "signal", "target": "unet_out"},
                {"action": "signal", "target": "text_encoders"},
                {"action": "signal", "target": "vae_out"},
            ],
        },
        "repo_id": {
            "label": "Repository ID",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": [""]},
                    "local": {"className": [""]},
                },
            },
        },
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "value": "float16",
            "postProcess": str_to_dtype,
        },
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "unet": {"label": "Denoise Model", "display": "input", "type": "diffusers_auto_model"},
        "vae": {"label": "VAE", "display": "input", "type": "diffusers_auto_model"},
        "lora_list": {"label": "Lora", "display": "input", "type": "custom_lora"},
        "text_encoders": {"label": "Text Encoders", "display": "output", "type": "diffusers_auto_models"},
        "unet_out": {"label": "Denoise Model", "display": "output", "type": "diffusers_auto_model"},
        "vae_out": {"label": "VAE", "display": "output", "type": "diffusers_auto_model"},
        "scheduler": {"label": "Scheduler", "display": "output", "type": "diffusers_auto_model"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.loader = None
        self.model_types_loaded = False

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        self.loader = None
        super().__del__()

    def set_filters(self, values, ref):
        # first time dynamically load the model_type options
        if not self.model_types_loaded:
            self.set_field_params("model_type", {"options": get_all_model_types()})
            self.model_types_loaded = True

        model_type = values.get("model_type", "")
        metadata = get_model_type_metadata(model_type)

        if metadata:
            default_repo = metadata["default_repo"]
            default_dtype = metadata["default_dtype"]
        else:
            # Fallback for empty or unknown model types
            default_repo = ""
            default_dtype = "float16"
        filters = [model_type]  # YiYi Notes: 1:1 between model_type <-> modular pipeline class

        self.set_field_params(
            "repo_id",
            {
                "default": {"source": "hub", "value": default_repo},
                "value": {"source": "hub", "value": default_repo},
                "fieldOptions": {
                    "filter": {"hub": {"className": filters}},
                },
            },
        )
        self.set_field_params("dtype", {"value": default_dtype})

    def execute(self, model_type, repo_id, device, dtype, unet=None, vae=None, lora_list=None):
        logger.debug(f"""
            ModelsLoader ({self.node_id}) received parameters:
            - repo_id: {repo_id}
            - dtype: {dtype}
            - device: {device}
            - unet: {unet}
            - vae: {vae}
        """)

        # TODO: add custom text encoders (depending on architecture)
        components_to_update = {}

        if unet:
            components_to_update.update(
                components.get_components_by_ids(ids=[unet["model_id"]], return_dict_with_names=True)
            )
        if vae:
            components_to_update.update(
                components.get_components_by_ids(ids=[vae["model_id"]], return_dict_with_names=True)
            )

        if isinstance(repo_id, dict):
            real_repo_id = repo_id.get("value", repo_id)
            _source = repo_id.get("source", "hub")  # TODO: do something when is local?
        else:
            real_repo_id = ""

        if real_repo_id == "":
            self.notify(
                "Please provide a valid Repository ID.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        if not components._auto_offload_enabled or components._auto_offload_device != device:
            components.enable_auto_cpu_offload(device=device)

        self.loader = ModularPipeline.from_pretrained(
            real_repo_id, components_manager=components, collection=self.node_id
        )

        if model_type == "DummyCustomPipeline":
            # update node param
            mellon_config = MellonPipelineConfig.load(real_repo_id)
            mellon_config.label = "Custom"

            # update repo_id for DummyCustomPipeline
            DummyCustomPipeline.repo_id = real_repo_id
            # register DummyCustomPipeline to MODULAR_REGISTRY
            MODULAR_REGISTRY.register(DummyCustomPipeline, mellon_config)

        else:
            DummyCustomPipeline.repo_id = None
            MODULAR_REGISTRY.register(DummyCustomPipeline, DUMMY_CUSTOM_PIPELINE_CONFIG)

        ALL_COMPONENTS = self.loader.pretrained_component_names

        text_node = self.loader.blocks.sub_blocks["text_encoder"].init_pipeline(real_repo_id)
        text_encoder_names = text_node.pretrained_component_names

        components_to_load = [c for c in ALL_COMPONENTS if c not in components_to_update]
        components_to_reload = []

        denoiser_options = ["unet", "transformer"]
        denoiser_name = next((option for option in denoiser_options if option in ALL_COMPONENTS), None)

        for comp_name in components_to_load:
            comp_spec = self.loader.get_component_spec(comp_name)
            if comp_spec.load_id != "null":
                comp_with_same_load_id = components._lookup_ids(load_id=comp_spec.load_id)
                same_comp_in_collection = []
                for comp_id in comp_with_same_load_id:
                    if isinstance(components.get_one(component_id=comp_id), torch.nn.Module):
                        comp_dtype = components.get_one(component_id=comp_id).dtype
                        if comp_dtype == dtype:
                            same_comp_in_collection.append(comp_id)
                    else:
                        same_comp_in_collection.append(comp_id)
                if not same_comp_in_collection:
                    components_to_reload.append(comp_name)

        self.loader.load_components(names=components_to_reload, torch_dtype=dtype)
        self.loader.update_components(**components_to_update)

        print(f" ModelsLoader: reloaded components: {components_to_reload}")
        print(f" ModelsLoader: updated components: {components_to_update.keys()}")

        if not lora_list:
            self.loader.unload_lora_weights()
        else:
            self.loader.unload_lora_weights()
            update_lora_adapters(self.loader, lora_list)

        # Construct loaded_components at the end after all modifications
        loaded_components = {
            "unet_out": node_get_component_info(node_id=self.node_id, manager=components, name=denoiser_name),
            "vae_out": node_get_component_info(node_id=self.node_id, manager=components, name="vae"),
            "text_encoders": {
                k: node_get_component_info(node_id=self.node_id, manager=components, name=k)
                for k in text_encoder_names
            },
            "scheduler": node_get_component_info(node_id=self.node_id, manager=components, name="scheduler"),
        }

        # add repo_id to all models info dicts
        for k, v in loaded_components.items():
            v["repo_id"] = real_repo_id

        logger.debug(f" ModelsLoader: Final component_manager state: {components}")

        return loaded_components
