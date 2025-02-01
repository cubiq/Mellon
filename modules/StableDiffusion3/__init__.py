from utils.torch_utils import device_list, default_device, str_to_dtype
from utils.hf_utils import list_local_models

MODULE_MAP = {
    'SD3PipelineLoader': {
        'label': 'SD3 Pipeline Loader',
        'category': 'loaders',
        'params': {
            'pipeline': {
                'label': 'SD3 Pipeline',
                'type': 'pipeline',
                'display': 'output',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'options': list_local_models(filters={"_class_name": r"Diffusion3Pipeline$"}),
                'display': 'autocomplete',
                'no_validation': True,
                'default': 'stabilityai/stable-diffusion-3.5-large',
            },
            'dtype': {
                'label': 'Dtype',
                'type': 'string',
                'options': ['auto', 'float32', 'float16', 'bfloat16'],
                'default': 'bfloat16',
                'postProcess': str_to_dtype,
            },
            'load_t5': {
                'label': 'Load T5 Encoder',
                'type': 'boolean',
                'default': True,
            },
            'transformer_in': {
                'label': 'Transformer',
                'type': 'SD3Transformer2DModel',
                'display': 'input',
            },
            'text_encoders_in': {
                'label': 'Text Encoders',
                'type': 'SD3TextEncoders',
                'display': 'input',
            },
            'vae_in': {
                'label': 'VAE',
                'type': 'VAE',
                'display': 'input',
            },
        },
    },

    'SD3TransformerLoader': {
        'label': 'SD3 Transformer loader',
        'description': 'Load the Transformer of an SD3 model',
        'category': 'loaders',
        'params': {
            'model': {
                'label': 'Transformer',
                'type': 'SD3Transformer2DModel',
                'display': 'output',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'options': list_local_models(),
                'display': 'autocomplete',
                'no_validation': True,
                'default': 'stabilityai/stable-diffusion-3.5-large',
            },
            'dtype': {
                'label': 'dtype',
                'options': ['auto', 'float32', 'float16', 'bfloat16'],
                'default': 'bfloat16',
                'postProcess': str_to_dtype,
            },
            'compile': {
                'label': 'Compile',
                'type': 'boolean',
                'default': False,
            },
        },
    },

    'SD3TextEncodersLoader': {
        'label': 'SD3 Text Encoders Loader',
        'description': 'Load both the CLIP and T5 Text Encoders',
        'category': 'loaders',
        'params': {
            'model': {
                'label': 'SD3 Encoders',
                'display': 'output',
                'type': 'SD3TextEncoders',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'options': list_local_models(),
                'display': 'autocomplete',
                'no_validation': True,
                'default': 'stabilityai/stable-diffusion-3.5-large',
            },
            'dtype': {
                'label': 'Dtype',
                'type': 'string',
                'options': ['auto', 'float32', 'float16', 'bfloat16'],
                'default': 'bfloat16',
            },
            'load_t5': {
                'label': 'Load T5 Encoder',
                'type': 'boolean',
                'default': True,
            },
        },
    },

    'SD3PromptEncoder': {
        'label': 'SD3 Prompt Encoder',
        'category': 'text-encoders',
        'params': {
            'text_encoders': {
                'label': 'SD3 Encoders | SD3 Pipeline',
                'display': 'input',
                'type': 'pipeline',
            },
            'embeds': {
                'label': 'Embeddings',
                'display': 'output',
                'type': 'SD3Embeddings',
            },
            'prompt': {
                'label': 'Prompt',
                'type': 'string',
                'display': 'textarea',
            },
            'negative_prompt': {
                'label': 'Negative Prompt',
                'type': 'string',
                'display': 'textarea',
            },
            'prompt_2': {
                'label': 'Prompt CLIP G',
                'type': 'string',
                'display': 'textarea',
                'group': { 'key': 'extra_prompts', 'label': 'Extra Prompts', 'display': 'collapse' },
            },
            'prompt_3': {
                'label': 'Prompt T5',
                'type': 'string',
                'display': 'textarea',
                'group': 'extra_prompts',
            },
            'negative_prompt_2': {
                'label': 'Negative CLIP G',
                'type': 'string',
                'display': 'textarea',
                'group': 'extra_prompts',
            },
            'negative_prompt_3': {
                'label': 'Negative T5',
                'type': 'string',
                'display': 'textarea',
                'group': 'extra_prompts',
            },
            'noise_clip': {
                'label': 'Clip Noise Pos',
                'type': 'float',
                'default': 0.0,
                'min': 0,
                'max': 1,
                'step': 0.01,
                'display': 'slider',
                'group': { 'key': 'noise', 'label': 'Noise', 'display': 'collapse' },
            },
            'noise_negative_clip': {
                'label': 'ClipNoise Neg',
                'type': 'float',
                'default': 0.0,
                'min': 0,
                'max': 1,
                'step': 0.01,
                'display': 'slider',
                'group': 'noise',
            },
            'noise_t5': {
                'label': 'T5 Noise Pos',
                'type': 'float',
                'default': 0.0,
                'min': 0,
                'max': 1,
                'step': 0.01,
                'display': 'slider',
                'group': 'noise',
            },
            'noise_negative_t5': {
                'label': 'T5 Noise Neg',
                'type': 'float',
                'default': 0.0,
                'min': 0,
                'max': 1,
                'step': 0.01,
                'display': 'slider',
                'group': 'noise',
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
        },
    },

    'SD3Sampler': {
        'label': 'SD3 Sampler',
        'category': 'samplers',
        'style': {
            'maxWidth': '360px',
        },
        'params': {
            'pipeline': {
                'label': 'Transformer | Pipeline',
                'display': 'input',
                'type': 'pipeline',
            },
            'prompt': {
                'label': 'Prompt',
                'display': 'input',
                'type': ['SD3Embeddings', 'embeddings'],
            },
            'latents_in': {
                'label': 'Latents',
                'display': 'input',
                'type': 'latent',
                'onChange': { 'action': 'disable', 'target': { 'connected': ['dimensions_group'], 'disconnected': ['denoise'] } },
            },
            'pipeline_out': {
                'label': 'Pipeline',
                'display': 'output',
                'type': 'pipeline',
            },
            'latents': {
                'label': 'Latents',
                'type': 'latent',
                'display': 'output',
            },
            'width': {
                'label': 'Width',
                'type': 'int',
                'display': 'text',
                'default': 1024,
                'min': 8,
                'max': 8192,
                'step': 8,
                'group': 'dimensions',
            },
            'height': {
                'label': 'Height',
                'type': 'int',
                'display': 'text',
                'default': 1024,
                'min': 8,
                'max': 8192,
                'step': 8,
                'group': 'dimensions',
            },
            'resolution_picker': {
                'label': 'Resolution',
                'display': 'ui',
                'type': 'dropdownIcon',
                'options': [
                    { 'label': ' 720×1280 (9:16)', 'value': [720, 1280] },
                    { 'label': ' 768×1344 (0.57)', 'value': [768, 1344] },
                    { 'label': ' 768×1280 (3:5)', 'value': [768, 1280] },
                    { 'label': ' 832×1152 (3:4)', 'value': [832, 1152] },
                    { 'label': '1024×1024 (1:1)', 'value': [1024, 1024] },
                    { 'label': ' 1152×832 (4:3)', 'value': [1152, 832] },
                    { 'label': ' 1280×768 (5:3)', 'value': [1280, 768] },
                    { 'label': ' 1344×768 (1.75)', 'value': [1344, 768] },
                    { 'label': ' 1280×720 (16:9)', 'value': [1280, 720] },
                ],
                'onChange': { 'action': 'set', 'target': ['width', 'height'] },
                'group': 'dimensions',
            },
            'seed': {
                'label': 'Seed',
                'type': 'int',
                'default': 0,
                'min': 0,
                'display': 'random',
            },
            'steps': {
                'label': 'Steps',
                'type': 'int',
                'default': 30,
                'min': 1,
                'max': 1000,
            },
            'cfg': {
                'label': 'Guidance',
                'type': 'float',
                'default': 5,
                'min': 0,
                'max': 100,
            },
            'denoise_range': {
                'label': 'Denoise Range',
                'type': 'float',
                'display': 'range',
                'default': [0, 1],
                'min': 0,
                'max': 1,
                'step': 0.01,
            },
            'scheduler': {
                'label': 'Scheduler',
                'display': 'select',
                'type': ['string', 'scheduler'],
                'options': {
                    'FlowMatchEulerDiscreteScheduler': 'Flow Match Euler Discrete',
                    'FlowMatchHeunDiscreteScheduler': 'Flow Match Heun Discrete',
                },
                'default': 'FlowMatchEulerDiscreteScheduler',
            },
            'shift': {
                'label': 'Shift',
                'type': 'float',
                'default': 3.0,
                'min': 0,
                'max': 12,
                'step': 0.05,
                'group': { 'key': 'scheduler', 'label': 'Scheduler options', 'display': 'collapse' },
            },
            'use_dynamic_shifting': {
                'label': 'Use dynamic shifting',
                'type': 'boolean',
                'default': False,
                'group': 'scheduler',
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
        },
    },
}

quantization_params = {
    'quantization': {
        'label': 'Quantization',
        'options': {
            'none': 'None',
            'bitsandbytes': 'BitsAndBytes',
            'quanto': 'Quanto',
            'torchao': 'TorchAO',
        },
        'default': 'none',
        'onChange': { 'action': 'show', 'target': { 'none': None, 'quanto': 'quanto_group', 'torchao': 'torchao_group', 'bitsandbytes': 'bitsandbytes_group' } },
    },

    # BitsAndBytes
    'bitsandbytes_weights': {
        'label': 'Weights',
        'options': ['8-bit', '4-bit'],
        'default': '4-bit',
        'group': { 'key': 'bitsandbytes', 'label': 'BitsAndBytes', 'display': 'group', 'direction': 'column' },
        'onChange': { 'action': 'show', 'target': { '4-bit': 'bitsandbytes_double_quant' } },
    },
    'bitsandbytes_double_quant': {
        'label': 'Use double quantization',
        'type': 'boolean',
        'default': True,
        'group': 'bitsandbytes'
    },

    # Quanto Quantization
    'quanto_weights': {
        'label': 'Weights',
        'options': ['int4', 'int8', 'float8'],
        'default': 'float8',
        'group': { 'key': 'quanto', 'label': 'Quanto Quantization', 'display': 'group', 'direction': 'column' },
    },
    'quanto_activations': {
        'label': 'Activations',
        'options': ['none', 'int4', 'int8', 'float8'],
        'default': 'none',
        'group': 'quanto'
    },
    'quanto_exclude': {
        'label': 'Exclude blocks',
        'description': 'Comma separated list of block names to exclude from quantization',
        'type': 'string',
        'default': 'proj_out',
        'group': 'quanto'
    },
    'quanto_device': {
        'label': 'Device',
        'type': 'string',
        'options': device_list,
        'default': default_device,
        'group': 'quanto'
    },

    # TorchAO Quantization
    'torchao_weights': {
        'label': 'Weights',
        'options': {
            'int8_weight_only': 'int8 weight',
            'int4_weight_only': 'int4 weight',
            'int8_dynamic_activation_int8_weight': 'int8 weight + activation',
            'fp6': 'fp6',
        },
        'default': 'int8_weight_only',
        'group': { 'key': 'torchao', 'label': 'TorchAO Quantization', 'display': 'group', 'direction': 'column' },
    },
    'torchao_individual_layers': {
        'label': 'Quantize each layer individually',
        'type': 'boolean',
        'default': False,
        'group': 'torchao'
    },
    'torchao_device': {
        'label': 'Device',
        'type': 'string',
        'options': device_list,
        'default': default_device,
        'group': 'torchao'
    },
}

MODULE_MAP['SD3TransformerLoader']['params'].update(quantization_params)
MODULE_MAP['SD3TextEncodersLoader']['params'].update(quantization_params)
