from typing import Dict, Any
from diffusers.modular_pipelines.mellon_node_utils import MellonNodeConfig, MellonParam
import logging

from diffusers.modular_pipelines import SequentialPipelineBlocks


logger = logging.getLogger("mellon")

# mellon nodes
SDXL_NODE_TYPES_PARAMS_MAP = {
    "controlnet": {
        "inputs": [
            "control_image",
            "controlnet_conditioning_scale",
            "control_guidance_start",
            "control_guidance_end",
            "height",
            "width",
        ],
        "model_inputs": [
            "controlnet",
        ],
        "outputs": [
            "controlnet_out",
        ],
        "block_names": [None],
    },
    "denoise": {
        "inputs": [
            "embeddings",
            "width",
            "height",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "image_latents",
            "strength",
            # custom adapters coming in as inputs
            "controlnet",
            # ip_adapter is optional and custom; include if available
            "ip_adapter",
        ],
        "model_inputs": [
            "unet",
            "guider",
            "scheduler",
        ],
        "outputs": [
            "latents",
            "latents_preview",
            "doc"
        ],
        "block_names": ["denoise"],
    },
    "vae_encoder": {
        "inputs": [
            "image",
            "width",
            "height",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "image_latents",
        ],
        "block_names": ["vae_encoder"],
    },
    "text_encoder": {
        "inputs": [
            "prompt",
            "negative_prompt",
        ],
        "model_inputs": [
            "text_encoders",
        ],
        "outputs": [
            "embeddings",
        ],
        "block_names": ["text_encoder"],
    },
    "decoder": {
        "inputs": [
            "latents",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "images",
        ],
        "block_names": ["decode"],
    },
}

QwenImage_NODE_TYPES_PARAMS_MAP = {
    "controlnet": {
        "inputs": [
            "control_image",
            "controlnet_conditioning_scale",
            "control_guidance_start",
            "control_guidance_end",
            "height",
            "width",
        ],
        "model_inputs": [
            "controlnet",
            "vae",
        ],
        "outputs": [
            "controlnet_out",
        ],
        "block_names": ["controlnet_vae_encoder"],
    },
    "denoise": {
        "inputs": [
            "embeddings",
            "width",
            "height",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "image_latents",
            "strength",
            "controlnet",
        ],
        "model_inputs": [
            "unet",
            "guider",
            "scheduler",
        ],
        "outputs": [
            "latents",
            "doc",
        ],
        "block_names": ["denoise"],
    },
    "vae_encoder": {
        "inputs": [
            "image",
            "width",
            "height",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "image_latents",
        ],
    },
    "text_encoder": {
        "inputs": [
            "prompt",
            "negative_prompt",
        ],
        "model_inputs": [
            "text_encoders",
        ],
        "outputs": [
            "embeddings",
        ],
    },
    "decoder": {
        "inputs": [
            "latents",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "images",
        ],
    },
}


QwenImageEdit_NODE_TYPES_PARAMS_MAP = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            "embeddings",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            MellonParam(name="image_latents", label="Image Latents", type="latents", display="input"),
        ],
        "model_inputs": [
            "unet",
            "guider",
            "scheduler",
        ],
        "outputs": [
            "latents",
            "doc",
        ],
        "block_names": ["denoise"],
    },
    "vae_encoder": {
        "inputs": [
            "image",
            "width",
            "height",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "image_latents",
        ],
    },
    "text_encoder": {
        "inputs": [
            "prompt",
            "negative_prompt",
            "image",
        ],
        "model_inputs": [
            "text_encoders",
        ],
        "outputs": [
            "embeddings",
        ],
    },
    "decoder": {
        "inputs": [
            "latents",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "images",
        ],
    },
}


QwenImageEditPlus_NODE_TYPES_PARAMS_MAP = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            "embeddings",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            MellonParam(name="image_latents", label="Image Latents", type="latents", display="input"),
        ],
        "model_inputs": [
            "unet",
            "guider",
            "scheduler",
        ],
        "outputs": [
            "latents",
            "doc",
        ],
        "block_names": ["denoise"],
    },
    "vae_encoder": {
        "inputs": [
            "image",
            "width",
            "height",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "image_latents",
        ],
    },
    "text_encoder": {
        "inputs": [
            "prompt",
            "negative_prompt",
            "image",
        ],
        "model_inputs": [
            "text_encoders",
        ],
        "outputs": [
            "embeddings",
        ],
    },
    "decoder": {
        "inputs": [
            "latents",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "images",
        ],
    },
}


Flux_NODE_TYPES_PARAMS_MAP = {
    "controlnet": {
        "inputs": [
            "control_image",
            "controlnet_conditioning_scale",
            "control_guidance_start",
            "control_guidance_end",
            "height",
            "width",
        ],
        "model_inputs": [
            "controlnet",
        ],
        "outputs": [
            "controlnet_out",
        ],
        "block_names": [None],
    },
    "denoise": {
        "inputs": [
            "embeddings",
            "width",
            "height",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "image_latents",
            "strength",
            "controlnet",
        ],
        "model_inputs": [
            "unet",
            "guider",
            "scheduler",
        ],
        "outputs": [
            "latents",
            "doc",
        ],
        "block_names": ["denoise"],
    },
    "vae_encoder": {
        "inputs": [
            "image",
            "width",
            "height",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "image_latents",
        ],
    },
    "text_encoder": {
        "inputs": [
            "prompt",
            "negative_prompt",
        ],
        "model_inputs": [
            "text_encoders",
        ],
        "outputs": [
            "embeddings",
        ],
    },
    "decoder": {
        "inputs": [
            "latents",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "images",
        ],
    },
}


FluxKontext_NODE_TYPES_PARAMS_MAP = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            "embeddings",
            "width",
            "height",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            MellonParam(name="image_latents", label="Image Latents", type="latents", display="input"),
        ],
        "model_inputs": [
            "unet",
            "guider",
            "scheduler",
        ],
        "outputs": [
            "latents",
            "doc",
        ],
        "block_names": ["denoise"],
    },
    "vae_encoder": {
        "inputs": [
            "image",
            "width",
            "height",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "image_latents",
        ],
    },
    "text_encoder": {
        "inputs": [
            "prompt",
            "negative_prompt",
        ],
        "model_inputs": [
            "text_encoders",
        ],
        "outputs": [
            "embeddings",
        ],
    },
    "decoder": {
        "inputs": [
            "latents",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "images",
        ],
    },
}

# Minimal modular registry for Mellon node configs
class ModularMellonNodeRegistry:
    """Registry mapping pipeline class to its node configs and metadata."""

    def __init__(self):
        self._registry = {}
        self._initialized = False

    def register(
        self, 
        pipeline_cls: type, 
        node_params: Dict[str, MellonNodeConfig],
        label: str = "",
        default_repo: str = "",
        default_dtype: str = ""
    ):
        if not self._initialized:
            _initialize_registry(self)

        model_type = pipeline_cls.__name__
        
        _meta_data = {
            "node_params": node_params,
            "label": label,
            "default_repo": default_repo,
            "model_type": model_type,
            "default_dtype": default_dtype,
        }
        
        self._registry[pipeline_cls] = _meta_data

    def get(self, pipeline_cls: type) -> MellonNodeConfig:
        if not self._initialized:
            _initialize_registry(self)
        return self._registry.get(pipeline_cls, None)

    def get_all(self) -> Dict[type, Dict[str, MellonNodeConfig]]:
        if not self._initialized:
            _initialize_registry(self)
        return self._registry


def _register_pipeline(
    pipeline_cls, 
    registry: ModularMellonNodeRegistry,
    params_map: Dict[str, Dict[str, Any]], 
    label: str = "",
    default_repo: str = "",
    default_dtype: str = "",
):
    """Register all node-type presets for a given pipeline class from a params map."""
    node_configs = {}
    for node_type, spec in params_map.items():
        if spec is None:
            node_config = None
        else:
            node_config = MellonNodeConfig(
                inputs=spec.get("inputs", []),
                model_inputs=spec.get("model_inputs", []),
                outputs=spec.get("outputs", []),
                blocks_names=spec.get("block_names", []),
                node_type=node_type,
            )
        node_configs[node_type] = node_config
    registry.register(
        pipeline_cls, 
        node_configs, 
        label=label, 
        default_repo=default_repo,
        default_dtype=default_dtype,
    )


def _initialize_registry(registry: ModularMellonNodeRegistry):
    """Initialize the registry and register all available pipeline configs."""
    print("Initializing registry")

    registry._initialized = True

    try:
        from diffusers import QwenImageModularPipeline
        _register_pipeline(
            QwenImageModularPipeline, 
            registry,
            QwenImage_NODE_TYPES_PARAMS_MAP, 
            label="Qwen Image",
            default_repo="Qwen/Qwen-Image",
            default_dtype="bfloat16",
        )
    except Exception as e:
        raise Exception(f"Failed to register QwenImageModularPipeline :{e}")

    try:
        from diffusers import QwenImageEditModularPipeline
        _register_pipeline(
            QwenImageEditModularPipeline, 
            registry,
            QwenImageEdit_NODE_TYPES_PARAMS_MAP, 
            label="Qwen Image Edit",
            default_repo="Qwen/Qwen-Image-Edit",
            default_dtype="bfloat16",
        )
    except Exception as e:
        raise Exception(f"Failed to register QwenImageEditModularPipeline :{e}")

    try:
        from diffusers import QwenImageEditPlusModularPipeline
        _register_pipeline(
            QwenImageEditPlusModularPipeline, 
            registry,
            QwenImageEditPlus_NODE_TYPES_PARAMS_MAP, 
            label="Qwen Image Edit Plus",
            default_repo="Qwen/Qwen-Image-Edit-2509",
            default_dtype="bfloat16",
        )
    except Exception as e:
        raise Exception(f"Failed to register QwenImageEditPlusModularPipeline :{e}")

    try:
        from diffusers import FluxModularPipeline
        _register_pipeline(
            FluxModularPipeline, 
            registry,
            Flux_NODE_TYPES_PARAMS_MAP,
            label="Flux",
            default_repo="black-forest-labs/FLUX.1-dev",
            default_dtype="bfloat16",
        )
    except Exception as e:
        raise Exception(f"Failed to register FluxModularPipeline :{e}")

    try:
        from diffusers import FluxKontextModularPipeline
        _register_pipeline(
            FluxKontextModularPipeline, 
            registry,
            FluxKontext_NODE_TYPES_PARAMS_MAP, 
            label="Flux Kontext",
            default_repo="black-forest-labs/FLUX.1-Kontext-dev",
            default_dtype="bfloat16",
        )
    except Exception as e:
        raise Exception(f"Failed to register FluxKontextModularPipeline :{e}")

    try:
        from diffusers import StableDiffusionXLModularPipeline
        _register_pipeline(
            StableDiffusionXLModularPipeline, 
            registry,
            SDXL_NODE_TYPES_PARAMS_MAP, 
            label="Stable Diffusion XL",
            default_repo="stabilityai/stable-diffusion-xl-base-1.0",
            default_dtype="float16",
        )
    except Exception as e:
        raise Exception(f"Failed to register StableDiffusionXLModularPipeline :{e}")


# Global singleton registry instance
MODULAR_REGISTRY = ModularMellonNodeRegistry()


def get_all_model_types() -> Dict[str, str]:

    """Get all registered model types with their labels for UI dropdowns.
    
    Returns:
        Dict mapping model type names (keys) to human-readable labels (values).
        
    Example output:
        {
            "": "",
            "StableDiffusionXLModularPipeline": "Stable Diffusion XL",
            "QwenImageModularPipeline": "Qwen Image",
            "QwenImageEditModularPipeline": "Qwen Image Edit",
            "FluxModularPipeline": "Flux",
            "FluxKontextModularPipeline": "Flux Kontext",
        }
    """

    registry = MODULAR_REGISTRY.get_all()
    all_labels = {}
    for _, meta_data in registry.items():
        all_labels[meta_data["model_type"]] = meta_data["label"]
    all_labels[""] = ""
    return all_labels


def get_model_type_signal_data() -> Dict[str, str]:
    """Get model type mapping for onSignal value actions.
    
    Returns a dict mapping model type names to themselves, used in onSignal
    to pass model type through from upstream nodes.
    
    Example:
        {
            "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
            "QwenImageModularPipeline": "QwenImageModularPipeline",
            "": "",
        }
    """
    registry = MODULAR_REGISTRY.get_all()
    # Get all registered model types and map them to themselves
    model_types = {}
    for _, meta_data in registry.items():
        model_type = meta_data["model_type"]
        model_types[model_type] = model_type
    
    # Add empty default
    model_types[""] = ""
    return model_types


def get_model_type_metadata(model_type: str) -> Dict[str, Any]:
    registry = MODULAR_REGISTRY.get_all()

    for _, meta_data in registry.items():
        if meta_data["model_type"] == model_type:
            return meta_data
    return None


def pipeline_class_to_mellon_node_config(pipeline_class, node_type=None):
    print(f" pipeline_class_to_mellon_node_config: {pipeline_class}, {node_type}")

    try:
        node_type_config = MODULAR_REGISTRY.get(pipeline_class)["node_params"][node_type]
        print(f" node_type_config: {node_type_config}")
    except Exception as e:
        logger.debug(f" Failed to load the node from {pipeline_class}: {e}")
        return None, None

    node_type_blocks = None
    pipeline = pipeline_class()

    if node_type_config is not None and node_type_config.blocks_names:
        blocks_dict = {
            name: block for name, block in pipeline.blocks.sub_blocks.items() if name in node_type_config.blocks_names
        }
        node_type_blocks = SequentialPipelineBlocks.from_blocks_dict(blocks_dict)

    return node_type_blocks, node_type_config