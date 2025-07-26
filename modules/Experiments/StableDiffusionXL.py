import torch
from PIL import Image
from mellon.NodeBase import NodeBase
from utils.torch_utils import str_to_dtype, DEVICE_LIST, DEFAULT_DEVICE
from utils.memory_menager import memory_flush
from mellon.config import CONFIG
from utils.huggingface import local_files_only, get_local_model_ids
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from .utils import get_clip_prompt_embeds, get_t5_prompt_embeds, upcast_vae

HF_TOKEN = CONFIG.hf['token']

class SDXLPipelineLoader(NodeBase):
    label = "SDXL Pipeline Loader"
    category = "loader"
    style = { "minWidth": 360 }
    params = {
        "pipeline": { "label": "SD3 Pipeline", "display": "output", "type": "pipeline" },
        "model_id": {
            "label": "Model",
            "display": "autocomplete",
            "type": "string",
            "default": "stabilityai/stable-diffusion-xl-base-1.0",
            "optionsSource": { "source": "hf_cache", "filter": { "className": "StableDiffusionXLPipeline" } },
            "fieldOptions": { "noValidation": True }
        },
        "dtype": {
            "label": "Dtype",
            "type": "string",
            "default": "bfloat16",
            "options": ['auto', 'float32', 'float16', 'bfloat16'],
        },
    }

    def execute(self, **kwargs):
        model_id = kwargs.get('model_id', 'stabilityai/stable-diffusion-xl-base-1.0')
        dtype = str_to_dtype(kwargs['dtype'])

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=local_files_only(model_id),
            variant="fp16",
        )

        return { "pipeline": pipeline }
