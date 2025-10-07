from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from .flux_layers import FLUX_LAYERS
from .t5_layers import T5_LAYERS
from .sd3_layers import SD3_LAYERS
from copy import deepcopy

def str_to_none(value, params):
    none_values = ['none', 'null', 'no', 'empty', 'ignore']
    return None if value in none_values else value

MODULE_PARSE = ['StableDiffusion3', 'VAE', 'FLUXKontext', 'StableDiffusionXL']

QUANT_FIELDS = {
    'bnb_group': {
        'label': 'BitsAndBytes quantization',
        'display': 'ui_group',
        'options': ['bnb_type', 'bnb_double_quant'],
        'default': 'none',
        'type': 'string',
        'style': {
            'borderTop': '1px solid rgba(255,255,255,0.1)',
            'paddingTop': 1,
        }
    },
        'bnb_type': {
            'label': 'Type',
            'options': ['8bit', '4bit'],
            'default': '4bit',
            'type': 'string',
            'onChange': {
                '8bit': [],
                '4bit': ['bnb_double_quant']
            }
        },
        'bnb_double_quant': {
            'label': 'Double Quantization',
            'type': 'boolean',
            'display': 'checkbox',
            'default': True,
        },

    'quanto_group': {
        'label': 'Quanto quantization',
        'display': 'ui_group',
        'options': ['quanto_weights', 'quanto_activations'],
        'default': 'none',
        'style': {
            'borderTop': '1px solid rgba(255,255,255,0.1)',
            'paddingTop': 1,
        }
    },
        'quanto_weights': {
            'label': 'Weights',
            'options': ['float8', 'int8', 'int4', 'int2'],
            'default': 'float8',
            'type': 'string',
        },
        'quanto_activations': {
            'label': 'Activations',
            'options': ['none', 'float8', 'int8'],
            'default': 'none',
            'type': 'string',
        },

    'torchao_group': {
        'label': 'TorchAO quantization',
        'display': 'ui_group',
        'options': ['torchao_quant_type'],
        'default': 'none',
        'style': {
            'borderTop': '1px solid rgba(255,255,255,0.1)',
            'paddingTop': 1,
        }
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
        },

    "quant_exclude": {
        "description": "Exclude layers from the quantization process.",
        "label": "Exclude Layers",
        "display": "autocomplete",
        "default": "",
        "options": [],
        "fieldOptions": {
            "multiple": True,
            "disableCloseOnSelect": True
        }
    },
    'quant_device': {
        'label': 'Quant Device',
        'options': DEVICE_LIST,
        'default': DEFAULT_DEVICE,
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
            'none': [],
            'bnb': ['bnb_group', 'quant_device'],
            'quanto': ['quanto_group', 'quant_exclude', 'quant_device'],
            'torchao': ['torchao_group', 'quant_exclude', 'quant_device']
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

MODULE_MAP = {
    'SD3TransformerLoader': {
        'label': "SD3 Transformer Loader",
        'category': "loader",
        'style': { "minWidth": 300 },
        'resizable': True,
        'params': {
            "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "default": { 'source': 'hub', 'value': 'stabilityai/stable-diffusion-3.5-large' },
            "fieldOptions": {
                "noValidation": True,
                "sources": ['hub', 'local'],
                "filter": {
                    "hub": { "className": ["SD3Transformer2DModel"] },
                    "local": { "id": r"sd3\.5" },
                },
            },
            },
            "dtype": {
                "label": "Dtype",
                "type": "string",
                "default": "bfloat16",
                "options": ['auto', 'float32', 'float16', 'bfloat16'],
            },
            **QUANT_SELECT,
            **deepcopy(QUANT_FIELDS),
            "fuse_qkv": {
                "description": "Improve performance at the cost of increased memory usage.",
                "label": "Fuse QKV projections",
                "type": "boolean",
                "default": False,
            },
            "compile": {
                "description": "Use Torch to compile the model for improved performance. Works only on supported platforms.",
                "label": "Compile",
                "type": "boolean",
                "default": False,
                "onChange": {
                    True: ['compile_mode', 'compile_fullgraph'],
                    False: []
                }
            },
            "compile_mode": {
                "label": "Mode",
                "type": "string",
                "default": "max-autotune",
                "options": ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
            },
            "compile_fullgraph": {
                "label": "Full Graph",
                "type": "boolean",
                "default": True,
            },
            "transformer": { "label": "Transformer", "display": "output", "type": "SD3Transformer2DModel" },
        }
    },

    'SD3TextEncodersLoader': {
        'description': "Load the CLIP and T5 Text Encoders",
        'label': 'SD3 Text Encoders Loader',
        'category': 'loader',
        'resizable': True,
        'params': {
            'encoders': {
                'label': 'SD3 Encoders',
                'display': 'output',
                'type': 'SD3TextEncoders',
            },
            'model_id': {
                "label": "Model",
                "display": "modelselect",
                "type": "string",
                "default": { 'source': 'hub', 'value': "stabilityai/stable-diffusion-3.5-large" },
                "fieldOptions": {
                    "noValidation": True,
                    "sources": ['hub', 'local'],
                    "filter": {
                        "hub": { "className": ["StableDiffusion3Pipeline"] },
                        "local": { "id": r"SD3\.5" },
                    },
                },
            },
            'dtype': {
                'label': 'Dtype',
                'type': 'string',
                'default': 'bfloat16',
                'options': ['auto', 'float32', 'float16', 'bfloat16'],
            },
            **QUANT_SELECT,
            **deepcopy(QUANT_FIELDS),
            'load_t5': { 'label': 'Load T5 Encoder', 'type': 'boolean', 'default': True },
        }
    },

    'FluxTransformerLoader': {
        'label': "FLUX Transformer Loader",
        'category': "loader",
        'style': { "minWidth": 300 },
        'resizable': True,
        'params': {
            "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "default": { 'source': 'hub', 'value': 'black-forest-labs/FLUX.1-dev' },
            "fieldOptions": {
                "noValidation": True,
                "sources": ['hub', 'local'],
                "filter": {
                    "hub": { "className": ["FluxTransformer2DModel"] },
                    "local": { "id": r"flux" },
                },
            },
            },
            "dtype": {
                "label": "Dtype",
                "type": "string",
                "default": "bfloat16",
                "options": ['auto', 'float32', 'float16', 'bfloat16'],
            },
            **QUANT_SELECT,
            **deepcopy(QUANT_FIELDS),
            "fuse_qkv": {
                "description": "Improve performance at the cost of increased memory usage.",
                "label": "Fuse QKV projections",
                "type": "boolean",
                "default": False,
            },
            "compile": {
                "description": "Use Torch to compile the model for improved performance. Works only on supported platforms.",
                "label": "Compile",
                "type": "boolean",
                "default": False,
                "onChange": {
                    True: ['compile_mode', 'compile_fullgraph'],
                    False: []
                }
            },
            "compile_mode": {
                "label": "Mode",
                "type": "string",
                "default": "max-autotune",
                "options": ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
            },
            "compile_fullgraph": {
                "label": "Full Graph",
                "type": "boolean",
                "default": True,
            },
            "transformer": { "label": "Transformer", "display": "output", "type": "FluxTransformer2DModel" },
        }
    },

    'FluxTextEncoderLoader': {
        'label': "FLUX Text Encoders Loader",
        'category': "loader",
        'params': {
            "model_id": {
                "label": "Model",
                "display": "modelselect",
                "type": "string",
                "default": { 'source': 'hub', 'value': 'black-forest-labs/FLUX.1-dev' },
                "fieldOptions": {
                    "noValidation": True,
                    "sources": ['hub'],
                    "filter": {
                        "hub": { "className": r"^Flux" },
                    },
                },
            },
            "dtype": {
                "label": "Dtype",
                "type": "string",
                "default": "bfloat16",
                "options": ['auto', 'float32', 'float16', 'bfloat16'],
            },
            **QUANT_SELECT,
            **deepcopy(QUANT_FIELDS),
            "t5": {
                "label": "T5 Encoder",
                "display": "input",
                "type": "T5EncoderModel",
            },
            "encoders": {
                "label": "Encoders",
                "display": "output",
                "type": "FluxTextEncoders",
            },
        }
    }
}

MODULE_MAP['SD3TextEncodersLoader']['params']['quant_exclude']['options'] = T5_LAYERS
MODULE_MAP['SD3TextEncodersLoader']['params']['quant_exclude']['default'] = T5_LAYERS[-1]
MODULE_MAP['SD3TransformerLoader']['params']['quant_exclude']['options'] = SD3_LAYERS
MODULE_MAP['SD3TransformerLoader']['params']['quant_exclude']['default'] = SD3_LAYERS[-1]

MODULE_MAP['FluxTransformerLoader']['params']['quant_exclude']['options'] = FLUX_LAYERS
MODULE_MAP['FluxTextEncoderLoader']['params']['quant_exclude']['options'] = T5_LAYERS
MODULE_MAP['FluxTextEncoderLoader']['params']['quant_exclude']['default'] = T5_LAYERS[-1]
MODULE_MAP['FluxTransformerLoader']['params']['quant_exclude']['default'] = FLUX_LAYERS[-1]
