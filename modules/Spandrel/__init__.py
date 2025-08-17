from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from utils.paths import list_files
from mellon.config import CONFIG
from mellon.modelstore import modelstore

MODULE_MAP = {
    "Upscaler": {
        "label": "Upscale with model",
        "category": "upscaler",
        "params": {
            "image": { "label": "Image", "type": "image", "display": "input" },
            "model_id": {
                "label": "Model",
                "display": "modelselect",
                "type": "string",
                "default": { 'source': 'local', 'value': next(iter(modelstore.get_local_ids(name="upscalers")), '') },
                "fieldOptions": {
                    "noValidation": True,
                    "sources": ['hub', 'local'],
                    "filter": {
                        "hub": {},
                        "local": { "id": r"^upscalers/" }
                    },
                },
            },

            # "model_id": {
            #     "label": "Model",
            #     "display": "autocomplete",
            #     "type": ["string", "filelist"],
            #     "options": [model.replace("upscalers/", "") for model in modelstore.get_local_ids(name="upscalers")],
            #     "fieldOptions": { "optionLabel": "label", "noValidation": True }
            # },
            "downscale": { "label": "Downscale", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01, "display": "slider", "description": "Post downscaling factor. After the image is upscaled, it is downscaled by this factor." },
            "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Image", "type": "image", "display": "output" },
        }
    },
}
