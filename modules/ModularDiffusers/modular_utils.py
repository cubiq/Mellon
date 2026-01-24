import logging
import threading
from typing import Any, Dict, Optional

from diffusers import Flux2KleinModularPipeline
from diffusers.modular_pipelines.mellon_node_utils import MellonParam, MellonPipelineConfig


logger = logging.getLogger("mellon")

SDXL_NODE_SPECS = {
    "controlnet": {
        "inputs": [
            MellonParam.control_image(),
            MellonParam.controlnet_conditioning_scale(),
            MellonParam.control_guidance_start(),
            MellonParam.control_guidance_end(),
            MellonParam.height(),
            MellonParam.width(),
        ],
        "model_inputs": [
            MellonParam.controlnet(),
        ],
        "outputs": [
            MellonParam.controlnet_bundle(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["control_image"],
        "required_model_inputs": ["controlnet"],
        "block_name": None,
    },
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.width(),
            MellonParam.height(),
            MellonParam.seed(),
            MellonParam.num_inference_steps(),
            MellonParam.guidance_scale(),
            MellonParam.image_latents_with_strength(),
            MellonParam.strength(),
            MellonParam.controlnet_bundle(display="input"),
            MellonParam.ip_adapter(),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
            MellonParam.controlnet_bundle(display="input"),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.latents_preview(),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            MellonParam.negative_prompt(),
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

SDXL_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=SDXL_NODE_SPECS,
    label="Stable Diffusion XL",
    default_repo="stabilityai/stable-diffusion-xl-base-1.0",
    default_dtype="float16",
)


# =============================================================================
# Qwen Image
# =============================================================================

QWEN_IMAGE_NODE_SPECS = {
    "controlnet": {
        "inputs": [
            MellonParam.control_image(),
            MellonParam.controlnet_conditioning_scale(),
            MellonParam.control_guidance_start(),
            MellonParam.control_guidance_end(),
            MellonParam.height(),
            MellonParam.width(),
        ],
        "model_inputs": [
            MellonParam.controlnet(),
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.controlnet_bundle(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["control_image"],
        "required_model_inputs": ["controlnet", "vae"],
        "block_name": "controlnet_vae_encoder",
    },
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.width(),
            MellonParam.height(),
            MellonParam.seed(),
            MellonParam.num_inference_steps(50),
            MellonParam.guidance_scale(4.5),
            MellonParam.image_latents_with_strength(),
            MellonParam.strength(),
            MellonParam.controlnet_bundle(display="input"),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
            MellonParam.controlnet_bundle(display="input"),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            MellonParam.negative_prompt(),
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

QWEN_IMAGE_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=QWEN_IMAGE_NODE_SPECS,
    label="Qwen Image",
    default_repo="Qwen/Qwen-Image-2512",
    default_dtype="bfloat16",
)


# =============================================================================
# Qwen Image Edit
# =============================================================================

QWEN_IMAGE_EDIT_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.seed(),
            MellonParam.num_inference_steps(40),
            MellonParam.guidance_scale(4.0),
            MellonParam.image_latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            MellonParam.negative_prompt(),
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt", "image"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

QWEN_IMAGE_EDIT_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=QWEN_IMAGE_EDIT_NODE_SPECS,
    label="Qwen Image Edit",
    default_repo="Qwen/Qwen-Image-Edit",
    default_dtype="bfloat16",
)


# =============================================================================
# Qwen Image Edit Plus
# =============================================================================

QWEN_IMAGE_EDIT_PLUS_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.seed(),
            MellonParam.num_inference_steps(40),
            MellonParam.guidance_scale(4.0),
            MellonParam.image_latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            MellonParam.negative_prompt(),
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt", "image"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

QWEN_IMAGE_EDIT_PLUS_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=QWEN_IMAGE_EDIT_PLUS_NODE_SPECS,
    label="Qwen Image Edit Plus",
    default_repo="Qwen/Qwen-Image-Edit-2511",
    default_dtype="bfloat16",
)

# =============================================================================
# Qwen Image Layered
# =============================================================================

QWEN_IMAGE_LAYERED_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.seed(),
            MellonParam.num_inference_steps(50),
            MellonParam.guidance_scale(4.0),
            MellonParam.layers(4),
            MellonParam.image_latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            MellonParam.negative_prompt(),
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt", "image"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

QWEN_IMAGE_LAYERED_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=QWEN_IMAGE_LAYERED_NODE_SPECS,
    label="Qwen Image Layered",
    default_repo="Qwen/Qwen-Image-Layered",
    default_dtype="bfloat16",
)

# =============================================================================
# Flux
# =============================================================================

FLUX_NODE_SPECS = {
    "controlnet": None,  # Not yet supported in Modular
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.width(),
            MellonParam.height(),
            MellonParam.seed(),
            MellonParam.num_inference_steps(28),
            MellonParam.guidance_scale(3.5),
            MellonParam.image_latents_with_strength(),
            MellonParam.strength(),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            # No negative_prompt - pipeline does not support this
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

FLUX_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=FLUX_NODE_SPECS,
    label="Flux",
    default_repo="black-forest-labs/FLUX.1-dev",
    default_dtype="bfloat16",
)


# =============================================================================
# Flux Kontext
# =============================================================================

FLUX_KONTEXT_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.seed(),
            MellonParam.num_inference_steps(28),
            MellonParam.guidance_scale(2.5),
            MellonParam.image_latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            # No negative_prompt
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

FLUX_KONTEXT_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=FLUX_KONTEXT_NODE_SPECS,
    label="Flux Kontext",
    default_repo="black-forest-labs/FLUX.1-Kontext-dev",
    default_dtype="bfloat16",
)

# =============================================================================
# Flux 2 Klein
# =============================================================================

FLUX_2_KLEIN_DISTILLED_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.width(),
            MellonParam.height(),
            MellonParam.seed(),
            MellonParam.num_inference_steps(4),
            MellonParam.guidance_scale(1.0),
            MellonParam.image_latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

FLUX_2_KLEIN_DISTILLED_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=FLUX_2_KLEIN_DISTILLED_NODE_SPECS,
    label="Flux 2 Klein Distilled",
    default_repo="black-forest-labs/FLUX.2-klein-4B",
    default_dtype="bfloat16",
)


# =============================================================================
# Z-Image
# =============================================================================

Z_IMAGE_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.width(),
            MellonParam.height(),
            MellonParam.seed(),
            MellonParam.num_inference_steps(9),
            MellonParam.guidance_scale(1.0),
            MellonParam.image_latents_with_strength(),
            MellonParam.strength(),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.guider(),
            MellonParam.scheduler(),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.image_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            # No negative_prompt - pipeline does not support this
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.images(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

Z_IMAGE_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=Z_IMAGE_NODE_SPECS,
    label="Z-Image",
    default_repo="Tongyi-MAI/Z-Image-Turbo",
    default_dtype="bfloat16",
)

# =============================================================================
# WAN
# =============================================================================

WAN_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MellonParam.embeddings(display="input"),
            MellonParam.width(832),
            MellonParam.height(480),
            MellonParam.seed(),
            MellonParam.num_inference_steps(50),
            MellonParam.guidance_scale(5.0),
            MellonParam.num_frames(81),
            MellonParam.first_frame_latents(display="input"),
            MellonParam.image_embeds(display="input"),
        ],
        "model_inputs": [
            MellonParam.unet(),
            MellonParam.scheduler(),
        ],
        "outputs": [
            MellonParam.latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.first_frame_latents(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "image_encoder": {
        "inputs": [
            MellonParam.image(),
        ],
        "model_inputs": [
            MellonParam.image_encoder(),
        ],
        "outputs": [
            MellonParam.image_embeds(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["image_encoder"],
        "block_name": "image_encoder",
    },
    "text_encoder": {
        "inputs": [
            MellonParam.prompt(),
            MellonParam.negative_prompt(),
        ],
        "model_inputs": [
            MellonParam.text_encoders(),
        ],
        "outputs": [
            MellonParam.embeddings(display="output"),
            MellonParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MellonParam.latents(display="input"),
        ],
        "model_inputs": [
            MellonParam.vae(),
        ],
        "outputs": [
            MellonParam.videos(),
            MellonParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

WAN_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs=WAN_NODE_SPECS,
    label="WAN",
    # default_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    default_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    default_dtype="bfloat16",
)


class DummyCustomPipeline:
    """Placeholder class used as registry key for custom pipelines."""

    repo_id = None

    def __new__(cls):
        from diffusers import ModularPipeline

        return ModularPipeline.from_pretrained(cls.repo_id, trust_remote_code=True)


DUMMY_CUSTOM_PIPELINE_CONFIG = MellonPipelineConfig(
    node_specs={}, label="Custom", default_repo="", default_dtype="bfloat16"
)


# Minimal modular registry for Mellon node configs
class ModularMellonNodeRegistry:
    """Registry mapping pipeline class to its config, including label, default_repo, default_dtype, and node_params."""

    def __init__(self):
        self._registry: Dict[type, MellonPipelineConfig] = {}
        self._initialized = False
        # Lock to prevent concurrent initialization races
        self._init_lock = threading.Lock()

    def register(self, pipeline_cls: type, config: MellonPipelineConfig):
        """Register a pipeline class with its config."""
        self._registry[pipeline_cls] = config

    def get(self, pipeline_cls: type) -> Optional[MellonPipelineConfig]:
        # Ensure only one thread/coroutine initializes the registry
        with self._init_lock:
            if not self._initialized:
                _initialize_registry(self)
        return self._registry.get(pipeline_cls, None)

    def get_all(self) -> Dict[type, MellonPipelineConfig]:
        # Ensure only one thread/coroutine initializes the registry
        with self._init_lock:
            if not self._initialized:
                _initialize_registry(self)
        return self._registry


def _initialize_registry(registry: ModularMellonNodeRegistry):
    """Initialize the registry and register all available pipeline configs."""
    logger.info("Initializing Mellon registry")

    # register DummyCustomPipeline with empty node specs
    registry.register(DummyCustomPipeline, DUMMY_CUSTOM_PIPELINE_CONFIG)

    try:
        from diffusers import StableDiffusionXLModularPipeline

        registry.register(StableDiffusionXLModularPipeline, SDXL_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register StableDiffusionXLModularPipeline: {e}")

    try:
        from diffusers import QwenImageModularPipeline

        registry.register(QwenImageModularPipeline, QWEN_IMAGE_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register QwenImageModularPipeline: {e}")

    try:
        from diffusers import QwenImageEditModularPipeline

        registry.register(QwenImageEditModularPipeline, QWEN_IMAGE_EDIT_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register QwenImageEditModularPipeline: {e}")

    try:
        from diffusers import QwenImageEditPlusModularPipeline

        registry.register(QwenImageEditPlusModularPipeline, QWEN_IMAGE_EDIT_PLUS_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register QwenImageEditPlusModularPipeline: {e}")

    try:
        from diffusers import QwenImageLayeredModularPipeline

        registry.register(QwenImageLayeredModularPipeline, QWEN_IMAGE_LAYERED_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register QwenImageLayeredModularPipeline: {e}")

    try:
        from diffusers import FluxModularPipeline

        registry.register(FluxModularPipeline, FLUX_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register FluxModularPipeline: {e}")

    try:
        from diffusers import FluxKontextModularPipeline

        registry.register(FluxKontextModularPipeline, FLUX_KONTEXT_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register FluxKontextModularPipeline: {e}")

    try:
        from diffusers import Flux2KleinModularPipeline

        registry.register(Flux2KleinModularPipeline, FLUX_2_KLEIN_DISTILLED_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register Flux2KleinModularPipeline: {e}")

    try:
        from diffusers import ZImageModularPipeline

        registry.register(ZImageModularPipeline, Z_IMAGE_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register ZImageModularPipeline: {e}")

    try:
        from diffusers import WanModularPipeline

        registry.register(WanModularPipeline, WAN_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register WanModularPipeline: {e}")

    registry._initialized = True


def _get_registry_instance():
    """Lazily import MODULAR_REGISTRY to avoid circular imports"""
    try:
        from . import MODULAR_REGISTRY
    except Exception as e:
        raise RuntimeError("MODULAR_REGISTRY not initialized.") from e
    return MODULAR_REGISTRY


def get_all_model_types() -> Dict[str, str]:
    """Get all registered model types with their labels for UI dropdowns.

    Returns:
        Dict mapping model type names (keys) to human-readable labels (values).

    Example output:
        {
            "": "",
            "StableDiffusionXLModularPipeline": "Stable Diffusion XL",
            "QwenImageModularPipeline": "Qwen Image",
            "FluxModularPipeline": "Flux",
        }
    """
    registry = _get_registry_instance().get_all()
    all_labels = {"": ""}
    for pipeline_cls, config in registry.items():
        model_type = pipeline_cls.__name__
        all_labels[model_type] = config.label
    return all_labels


# YiYi notes: not used for now
def get_model_type_signal_data() -> Dict[str, str]:
    """Get model type mapping for onSignal value actions.

    Returns a dict mapping model type names to themselves, used in onSignal
    to pass model type through from upstream nodes.
    """
    registry = _get_registry_instance().get_all()
    model_types = {"": ""}
    for pipeline_cls, _ in registry.items():
        model_type = pipeline_cls.__name__
        model_types[model_type] = model_type
    return model_types


def get_model_type_metadata(model_type: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a model type.

    Returns dict with model_type, label, default_repo, default_dtype, node_params.
    """
    registry = _get_registry_instance().get_all()
    for pipeline_cls, config in registry.items():
        if pipeline_cls.__name__ == model_type:
            return {
                "model_type": model_type,
                "label": config.label,
                "default_repo": config.default_repo,
                "default_dtype": config.default_dtype,
                "node_params": config.node_params,
            }
    return None


def pipeline_class_to_mellon_node_config(pipeline_class, node_type=None):
    """Get the block and mellon node params for a pipeline class and node type."""
    config = _get_registry_instance().get(pipeline_class)
    if config is None:
        logger.debug(f"Failed to load config for {pipeline_class}")
        return None, None

    node_params = config.node_params.get(node_type)

    node_type_blocks = None
    if node_params is not None and node_params.get("block_name"):
        # patch to use only distilled klein blocks
        if pipeline_class == Flux2KleinModularPipeline:
            pipeline = pipeline_class(config_dict={"is_distilled": True})
        else:
            pipeline = pipeline_class()

        node_type_blocks = pipeline.blocks.sub_blocks[node_params["block_name"]]

    return node_type_blocks, node_params
