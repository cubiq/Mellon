from utils.huggingface import get_local_model_ids
from utils.torch_utils import DEFAULT_DEVICE, device_list, str_to_dtype

MODULE_MAP = {
    "ModelsLoader": {
        "description": "",
        "label": "Load Models",
        "category": "Modular Diffusers",
        "resizable": "True",
        "params": {
            "repo_id": {
                "label": "Repository ID",
                "display": "autocomplete",
                "type": "string",
                "options": get_local_model_ids(
                    class_name="StableDiffusionXLModularPipeline"
                ),
                "fieldOptions": {"noValidation": True},
            },
            "dtype": {
                "label": "dtype",
                "options": ["float32", "float16", "bfloat16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "device": {
                "label": "Device",
                "type": "string",
                "default": DEFAULT_DEVICE,
                "options": device_list(),
            },
            "unet": {
                "label": "Unet",
                "display": "input",
                "type": "diffusers_auto_model",
            },
            "vae": {
                "label": "VAE",
                "display": "input",
                "type": "diffusers_auto_model",
            },
            "text_encoders": {
                "label": "Text Encoders",
                "display": "output",
                "type": "text_encoders",
            },
            "unet_out": {
                "label": "UNet",
                "display": "output",
                "type": "diffusers_auto_model",
            },
            "vae_out": {
                "label": "VAE",
                "display": "output",
                "type": "diffusers_auto_model",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "output",
                "type": "scheduler",
            },
        },
    }
}
