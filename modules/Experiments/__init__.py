from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from utils.huggingface import get_local_model_ids
from mellon.modelstore import modelstore
from .flux_layers import FLUX_LAYERS

MODULE_PARSE = ['StableDiffusion3', 'VAE', 'FLUXKontext', 'StableDiffusionXL']

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
    'quanto_type': {
        'label': 'Type',
        'options': ['float8', 'int8', 'int4', 'int2'],
        'default': 'float8',
        'type': 'string',
    },
    'torchao_quant_type': {
        'label': 'Quant Type',
        'options': [
            'int4wo', 'int4dq', 'int8wo', 'int8dq',
            'uint1wo', 'uint2wo', 'uint3wo', 'uint4wo', 'uint5wo', 'uint6wo', 'uint7wo',
            'float8wo_e5m2', 'float8wo_e4m3', 'float8dq_e4m3', 'float8dq_e4m3_tensor', 'float8dq_e4m3_row',
            'fp3_e1m1', 'fp3_e2m0', 'fp4_e1m2', 'fp4_e2m1', 'fp4_e3m0', 'fp5_e1m3', 'fp5_e2m2',
            'fp5_e3m1', 'fp5_e4m0', 'fp6_e1m4', 'fp6_e2m3', 'fp6_e3m2', 'fp6_e4m1', 'fp6_e5m0',
            'fp7_e1m5', 'fp7_e2m4', 'fp7_e3m3', 'fp7_e4m2', 'fp7_e5m1', 'fp7_e6m0'
        ],
        'default': 'float8wo_e4m3',
        'type': 'string',
    }
}

QUANT_SELECT = {
    'quantization': {
        'label': 'Quantization',
        'type': 'string',
        'options': { 'none': 'None', 'bnb': 'BitsAndBytes', 'quanto': 'Optimum Quanto', 'torchao': 'TorchAO' },
        'default': 'none',
        'onChange': {
            'bnb': ['bnb_type', 'bnb_double_quant'],
            'quanto': ['quanto_type'],
            'torchao': ['torchao_quant_type']
        },
    },
}

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    #(800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    #(1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]