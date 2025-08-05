from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from utils.huggingface import get_local_model_ids
from mellon.modelstore import modelstore

MODULE_PARSE = ['StableDiffusion3', 'VAE', 'FLUXKontext', 'StableDiffusionXL', 'QwenImage']

MODULE_MAP = {
    "SD3PipelineLoader": {
        "description": "Load a Stable Diffusion 3 pipeline",
        "label": "SD3 Pipeline Loader",
        "category": "loader",
        "style": { "minWidth": 360 },
        "params": {
            "pipeline": { "label": "SD3 Pipeline", "display": "output", "type": "pipeline" },
            "model_id": {
                "label": "Model",
                "display": "autocomplete",
                "type": "string",
                'options': modelstore.get_hf_ids(class_name="StableDiffusion3Pipeline"),
                "default": "stabilityai/stable-diffusion-3.5-large",
                "fieldOptions": { "noValidation": True }
            },
            "dtype": {
                "label": "Dtype",
                "type": "string",
                "default": "bfloat16",
                "options": ['auto', 'float32', 'float16', 'bfloat16'],
            },
            "load_t5": { "label": "Load T5 Encoder", "type": "boolean", "default": True },
            "text_encoders": { "label": "Text Encoders", "display": "input", "type": "SD3TextEncoders", "onChange": { True: [], False: ['load_t5']} },
            "transformer": { "label": "Transformer", "display": "input", "type": "SD3Transformer2DModel" },
        }
    },

    'SD3TextEncodersLoader': {
        'label': 'SD3 Text Encoders Loader',
        'description': 'Load the CLIP and T5 Text Encoders',
        'category': 'loader',
        'params': {
            'encoders': {
                'label': 'SD3 Encoders',
                'display': 'output',
                'type': 'SD3TextEncoders',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'default': 'stabilityai/stable-diffusion-3.5-large',
                'options': modelstore.get_hf_ids(class_name="StableDiffusion3Pipeline"),
                'display': 'autocomplete',
                'fieldOptions': { "noValidation": True, "model_loader": True },
            },
            'dtype': {
                'label': 'Dtype',
                'type': 'string',
                'default': 'bfloat16',
                'options': ['auto', 'float32', 'float16', 'bfloat16'],
            },
            'load_t5': { 'label': 'Load T5 Encoder', 'type': 'boolean', 'default': True },
        },
    },
}

QUANT_FIELDS = {
    'bnb_type': {
        'label': 'Type',
        'options': ['8bit', '4bit'],
        'default': '4bit',
        'type': 'string',
    },
    'bnb_double_quant': {
        'label': 'Double Quant',
        'type': 'boolean',
        'default': True,
    },
}

QUANT_SELECT = {
    'quantization': {
        'label': 'Quantization',
        'type': 'string',
        'options': { 'none': 'None', 'bnb': 'BitsAndBytes' },
        'default': 'none',
        'onChange': {
            'bnb': ['bnb_type', 'bnb_double_quant'],
        },
    },
}

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]