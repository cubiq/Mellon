import os
import logging
logger = logging.getLogger('mellon')
from mellon.NodeBase import NodeBase
from utils.torch_utils import str_to_dtype, DEVICE_LIST, DEFAULT_DEVICE, IS_CUDA
from utils.huggingface import get_model_class
from mellon.config import CONFIG
from diffusers import FluxTransformer2DModel, AutoencoderKL
from modules.Experiments import QUANT_FIELDS, QUANT_SELECT, PREFERRED_KONTEXT_RESOLUTIONS
from utils.quantization import getQuantizationConfig, quantize
from utils.image import fit as image_fit
from utils.memory_menager import memory_flush
import torch
import importlib
import os
from .flux_layers import FLUX_LAYERS

HF_TOKEN = CONFIG.hf['token']
MODELS_DIR = CONFIG.paths['models']
ONLINE_STATUS = CONFIG.hf['online_status']

class FluxTransformerLoader(NodeBase):
    label = "FLUX Transformer Loader"
    category = "loader"
    style = { "minWidth": 300 }
    resizable = True
    params = {
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
        **QUANT_FIELDS,
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

    def execute(self, **kwargs):
        model_id = kwargs.get('model_id', { 'source': 'hub', 'value': 'black-forest-labs/FLUX.1-dev'})
        model_id = model_id.get('value', 'black-forest-labs/FLUX.1-dev') if isinstance(model_id, dict) else model_id

        dtype = str_to_dtype(kwargs['dtype'])

        quantization = kwargs.get('quantization', 'none')
        quantization = None if quantization == 'none' else quantization
        quantization = 'gguf' if model_id.lower().endswith('.gguf') else quantization

        fuse_qkv = kwargs.get('fuse_qkv', False)

        compile = kwargs.get('compile', False)
        compile_mode = kwargs.get('compile_mode', 'default')
        compile_fullgraph = kwargs.get('compile_fullgraph', False)

        config = {
            'torch_dtype': dtype,
            'token': HF_TOKEN,
            'subfolder': "transformer"
        }

        loaderCallback = FluxTransformer2DModel.from_pretrained
        if quantization == 'gguf':
            from diffusers import GGUFQuantizationConfig
            config['quantization_config'] = GGUFQuantizationConfig(compute_dtype=dtype)
            loaderCallback = FluxTransformer2DModel.from_single_file
            if 'kontext' in model_id.lower():
                # workaround for an issue with GGUF Kontext models not having the correct in_channels
                # https://github.com/huggingface/diffusers/issues/11839
                config['in_channels'] = 64
        elif quantization == 'bnb':
            config['quantization_config'] = getQuantizationConfig(quantization, **kwargs)

        transformer = self.graceful_model_loader(loaderCallback, model_id, config)

        if quantization == 'torchao' or quantization == 'quanto':
            quant_device = kwargs.get('quant_device', None)
            transformer = self.mm_exec(lambda: quantize(transformer, quantization, **kwargs), quant_device, models=[transformer])
            memory_flush()

        # for name, module in transformer.named_modules():
        #     with open("transformer_modules.txt", "a") as f:
        #         f.write(f'"{name}",\n')

        if fuse_qkv and hasattr(transformer, 'fuse_qkv_projections'):
            transformer.fuse_qkv_projections()

        if compile:
            transformer = torch.compile(transformer, mode=compile_mode, fullgraph=compile_fullgraph)

        self.mm_add(transformer, priority=3)

        return { "transformer": transformer }

class FluxTextEncoderLoader(NodeBase):
    label = "FLUX Text Encoders Loader"
    category = "loader"
    params = {
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
        **QUANT_FIELDS,
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

    def execute(self, **kwargs):
        from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast, T5EncoderModel

        model_id = kwargs.get('model_id', { 'source': 'hub', 'value': 'black-forest-labs/FLUX.1-dev'})
        model_id = model_id.get('value', 'black-forest-labs/FLUX.1-dev') if isinstance(model_id, dict) else model_id
        dtype = str_to_dtype(kwargs.get('dtype', 'bfloat16'))
        quantization = kwargs.get('quantization', 'none')
        quantization = None if quantization == 'none' else quantization

        t5 = kwargs.get('t5', None)

        quant_config = None
        if not t5 and quantization == 'bnb':
            quant_config = getQuantizationConfig(quantization, **kwargs)

        config = {
            "torch_dtype": dtype,
            "token": HF_TOKEN,
        }

        text_encoder = self.graceful_model_loader(CLIPTextModel, model_id, {**config, "subfolder": "text_encoder"})
        text_encoder_2 = t5 or self.graceful_model_loader(T5EncoderModel, model_id, {**config, "subfolder": "text_encoder_2", "quantization_config": quant_config})
        tokenizer = self.graceful_model_loader(CLIPTokenizer, model_id, {**config, "subfolder": "tokenizer"})
        tokenizer_2 = self.graceful_model_loader(T5TokenizerFast, model_id, {**config, "subfolder": "tokenizer_2"})

        self.mm_add(text_encoder, priority=1)
        if not t5:
            self.mm_add(text_encoder_2, priority=1)

            if quantization == 'torchao' or quantization == 'quanto':
                quant_device = kwargs.get('quant_device', None)
                text_encoder_2 = self.mm_exec(lambda: quantize(text_encoder_2, quantization, **kwargs), quant_device, models=[text_encoder_2])
                memory_flush()
        
        #print(dict(text_encoder_2.named_parameters()).keys())
        #print(dict(text_encoder_2.named_modules()).keys())

        return { "encoders": { "text_encoder": text_encoder, "text_encoder_2": text_encoder_2, "tokenizer": tokenizer, "tokenizer_2": tokenizer_2 } }

class FluxPipelineLoader(NodeBase):
    label = "FLUX Pipeline Loader"
    category = "loader"
    params = {
        "pipeline": { "label": "FLUX Pipeline", "display": "output", "type": "pipeline" },
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
        "transformer": { "label": "Transformer", "display": "input", "type": "FluxTransformer2DModel" },
        "encoders": { "label": "Encoders", "display": "input", "type": "FluxTextEncoders" },
    }

    def execute(self, **kwargs):
        model_id = kwargs.get('model_id', { 'source': 'hub', 'value': 'black-forest-labs/FLUX.1-dev'})
        source = model_id.get('source', 'hub')
        model_id = model_id.get('value', 'black-forest-labs/FLUX.1-dev') if isinstance(model_id, dict) else model_id
        dtype = str_to_dtype(kwargs['dtype'])
        transformer = kwargs.get('transformer', None)
        encoders = kwargs.get('encoders', None)

        config = {
            'torch_dtype': dtype,
            'token': HF_TOKEN,
        }

        if transformer:
            config['transformer'] = transformer
        if encoders:
            config['text_encoder'] = encoders['text_encoder']
            config['text_encoder_2'] = encoders['text_encoder_2']
            config['tokenizer'] = encoders['tokenizer']
            config['tokenizer_2'] = encoders['tokenizer_2']

        fluxClass = get_model_class(model_id)

        # the repository is not cached, we infer the class name from the model_id
        if not fluxClass:
            if 'kontext' in model_id.lower():
                fluxClass = 'FluxKontextPipeline'
            else:
                fluxClass = 'FluxPipeline'

        try:
            diffusers_mod = importlib.import_module("diffusers")
            FluxPipeline = getattr(diffusers_mod, fluxClass)
        except Exception:
            logger.error(f"Error loading Pipeline class {fluxClass} for model {model_id}.")
            return None

        pipeline = self.graceful_model_loader(FluxPipeline, model_id, config)

        self.mm_add(pipeline.vae, priority=2)

        if not encoders:
            self.mm_add(pipeline.text_encoder, priority=1)
            self.mm_add(pipeline.text_encoder_2, priority=1)

        if not transformer:
            self.mm_add(pipeline.transformer, priority=3)

        return { "pipeline": pipeline }

class FluxTextEncoder(NodeBase):
    label = "FLUX Text Encoder"
    category = "embedding"
    resizable = True
    params = {
        "pipeline": { "label": "FLUX Pipeline", "display": "input", "type": ["pipeline", "FluxTextEncoders"] },
        "embeds": { "label": "Embeddings", "display": "output", "type": "embedding" },
        "prompt": { "label": "Prompt", "type": "text" },
        #"negative_prompt": { "label": "Negative Prompt", "type": "text" },
        "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get('pipeline', None)
        prompt = kwargs.get('prompt', '')
        device = kwargs.get('device', DEFAULT_DEVICE)

        try:
            pipelineClass = pipeline.__class__.__name__ or 'FluxPipeline'
            diffusers_mod = importlib.import_module("diffusers")
            FluxPipeline = getattr(diffusers_mod, pipelineClass)
        except Exception as e:
            logger.error(f"Error loading pipeline class {pipelineClass}: {e}")
            return None

        work_pipe = FluxPipeline.from_pretrained(
            pipeline.config._name_or_path,
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            transformer=None,
            vae=None,
            local_files_only=True,
        )

        pooled_prompt_embeds = self.mm_exec(
            lambda: work_pipe._get_clip_prompt_embeds(prompt=prompt, device=device, num_images_per_prompt=1),
            device,
            models=[work_pipe.text_encoder],
        )

        prompt_embeds = self.mm_exec(
            lambda: work_pipe._get_t5_prompt_embeds(prompt=prompt, device=device, num_images_per_prompt=1),
            device,
            models=[work_pipe.text_encoder_2],
        )

        (
            prompt_embeds,
            pooled_prompt_embeds,
            _,
        ) = self.mm_exec(
            lambda: work_pipe.encode_prompt(None, None, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, device=device),
            device,
            models=[work_pipe.text_encoder, work_pipe.text_encoder_2],
        )
        del work_pipe

        return { "embeds": (prompt_embeds, pooled_prompt_embeds) }

class FluxKontextSampler(NodeBase):
    label = "FLUX Kontext Sampler"
    category = "sampler"
    params = {
        "pipeline": { "label": "FLUX Pipeline", "display": "input", "type": "pipeline" },
        "embeds": { "label": "Embeddings", "display": "input", "type": "embedding" },
        "image": { "label": "Image", "display": "input", "type": "image" },
        "latents": { "label": "Latents", "display": "output", "type": "latent" },
        "seed": { "label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295 },
        "resolution": {
            "label": "Resolution",
            "type": "string",
            "default": "1024x1024",
            "options": [f"{w}x{h}" for w, h in PREFERRED_KONTEXT_RESOLUTIONS],
        },
        "steps": { "label": "Steps", "type": "int", "default": 25, "min": 1, "max": 100, "step": 1 },
        "cfg": { "label": "Guidance Scale", "type": "float", "default": 2.5, "min": 0.0, "max": 15.0, "step": 0.1 },
        "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },

    }
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.curr_image = None
        self.curr_image_latents = None

    def execute(self, **kwargs):
        pipeline = kwargs.get('pipeline', None)
        prompt_embeds, pooled_prompt_embeds = kwargs.get('embeds', (None, None))
        cfg = kwargs['cfg']
        image = kwargs['image']
        resolution = kwargs.get('resolution', '1024x1024')
        width, height = [int(x) for x in resolution.split('x')]
        device = kwargs.get('device', DEFAULT_DEVICE)
        seed = kwargs.get('seed', 0)
        steps = kwargs.get('steps', 25)
        image = image[0] if isinstance(image, list) else image

        generator = torch.Generator(device=device).manual_seed(seed)

        try:
            pipelineClass = pipeline.__class__.__name__ or 'FluxKontextPipeline'
            diffusers_mod = importlib.import_module("diffusers")
            FluxPipeline = getattr(diffusers_mod, pipelineClass)
        except Exception as e:
            logger.error(f"Error loading pipeline class {pipelineClass}: {e}")
            return None

        if self.curr_image is image:
            image_latents = self.curr_image_latents
        else:
            encode_pipe = FluxPipeline.from_pretrained(
                pipeline.config._name_or_path,
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
                transformer=None,
                vae=pipeline.vae,
                local_files_only=True,
            )

            def encode_image(pipe, image, device, generator):
                dtype = pipe.vae.dtype
                image = image.convert('RGB')
                w, h = image.size
                aspect_ratio = w / h
                _, ref_w, ref_h = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
                image = image_fit(image, ref_w, ref_h, resample='LANCZOS')
                image = pipe.image_processor.preprocess(image)
                image = image.to(device, dtype=dtype)
                image_latents = pipe._encode_vae_image(image, generator)
                image_latents = image_latents.to('cpu').detach().clone()
                del pipe, image
                return image_latents

            image_latents = self.mm_exec(
                lambda: encode_image(encode_pipe, image, device, generator),
                device,
                models=[encode_pipe.vae],
            )

            self.curr_image = image
            self.curr_image_latents = image_latents
            del encode_pipe
            memory_flush()

        dummy_vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            latent_channels=16,
        )

        sampling_config = {
            'image': image_latents,
            'generator': generator,
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'width': width,
            'height': height,
            'guidance_scale': cfg,
            'num_inference_steps': steps,
            'output_type': "latent",
            'callback_on_step_end': self.pipe_callback,
        }

        sampling_pipeline = FluxPipeline.from_pretrained(
            pipeline.config._name_or_path,
            transformer=pipeline.transformer,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            local_files_only=True,
            vae=dummy_vae,
        )

        def sampling(pipe, config, device):
            pipe.vae.to(device)
            config['image'] = config['image'].to(device, dtype=pipe.transformer.dtype)
            config['prompt_embeds'] = config['prompt_embeds'].to(device, dtype=pipe.transformer.dtype)
            config['pooled_prompt_embeds'] = config['pooled_prompt_embeds'].to(device, dtype=pipe.transformer.dtype)

            latents = pipe(**config).images
            config['image'] = config['image'].to('cpu')
            config['prompt_embeds'] = config['prompt_embeds'].to('cpu')
            config['pooled_prompt_embeds'] = config['pooled_prompt_embeds'].to('cpu')
            del pipe, config
            return latents.to('cpu').detach().clone()

        latents = self.mm_exec(
            lambda: sampling(sampling_pipeline, sampling_config, device),
            device,
            models=[sampling_pipeline.transformer],
        )
        del sampling_pipeline, dummy_vae, image_latents, prompt_embeds, pooled_prompt_embeds

        return { "latents": (latents, (height, width)) }


class NunchakuFluxTransformerLoader(NodeBase):
    label = "Nunchaku FLUX Transformer Loader"
    category = "loader"
    params = {
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "options": ["nunchaku-tech/nunchaku-flux.1-dev", "nunchaku-tech/nunchaku-flux.1-kontext-dev", "nunchaku-tech/nunchaku-flux.1-krea-dev"],
            "default": { 'source': 'hub', 'value': 'nunchaku-tech/nunchaku-flux.1-dev' },
            "fieldOptions": {
                "noValidation": True,
                "sources": ['hub', 'local'],
                "filter": {
                    "hub": { "id": r"nunchaku.*-flux\.1" },
                    "local": { "id": r"svdq-.*-flux\.1" }
                },
            }
        },
        "fp16_attn": {
            "label": "Enable FP16 Attention",
            "type": "bool",
            "default": IS_CUDA,
            "description": "Use Nunchaku's FP16 attention implementation for better performance. Only available on NVIDIA devices from series 30xx and above.",
        },
        "dtype": {
            "label": "Dtype",
            "type": "string",
            "default": "bfloat16",
            "options": ['auto', 'float32', 'float16', 'bfloat16'],
        },
        "transformer": { "label": "Transformer", "display": "output", "type": "FluxTransformer2DModel" }
    }

    def execute(self, **kwargs):
        from nunchaku import NunchakuFluxTransformer2dModel
        from nunchaku.utils import get_precision

        model_id = kwargs.get('model_id', { 'source': 'hub', 'value': 'nunchaku-tech/nunchaku-flux.1-dev'})
        source = model_id.get('source', 'hub')
        model_id = model_id.get('value', 'nunchaku-tech/nunchaku-flux.1-dev')
        dtype = str_to_dtype(kwargs.get('dtype', 'bfloat16'))
        fp16_attn = kwargs.get('fp16_attn', False)

        if source == 'local':
            if not os.path.isabs(model_id):
                model_id = os.path.join(MODELS_DIR, model_id)
            if not os.path.exists(model_id):
                raise FileNotFoundError(f"Local model {model_id} not found.")
        else:
            if not model_id.endswith('.safetensors'):
                filename = model_id.split('/')[-1].replace('nunchaku-', f"svdq-{get_precision()}_r32-") + '.safetensors'
                model_id = f"{model_id}/{filename}"

        transformer = self.graceful_model_loader(NunchakuFluxTransformer2dModel, model_id, { "torch_dtype": dtype, "token": HF_TOKEN })

        if fp16_attn and hasattr(transformer, 'set_attention_impl'):
            transformer.set_attention_impl("nunchaku-fp16")

        self.mm_add(transformer, priority=3)

        return { "transformer": transformer }

class NunchakuT5EncoderLoader(NodeBase):
    label = "Nunchaku T5 Encoder Loader"
    category = "loader"
    params = {
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "default": { 'source': 'hub', 'value': 'nunchaku-tech/nunchaku-t5' },
            "fieldOptions": {
                "noValidation": True,
                "sources": ['hub', 'local'],
                "filter": {
                    "hub": { "id": r"nunchaku-t5" },
                    "local": { "id": r"awq-.*-t5xxl" }
                },
            },
        },
        "dtype": {
            "label": "Dtype",
            "type": "string",
            "default": "bfloat16",
            "options": ['auto', 'float32', 'float16', 'bfloat16'],
        },
        "t5": {
            "label": "T5 Encoder",
            "display": "output",
            "type": "T5EncoderModel",
        },
    }

    def execute(self, **kwargs):
        from nunchaku import NunchakuT5EncoderModel

        model_id = kwargs.get('model_id', { 'source': 'hub', 'value': 'nunchaku-tech/nunchaku-t5'})
        source = model_id.get('source', 'hub')
        model_id = model_id.get('value', 'nunchaku-tech/nunchaku-t5')
        dtype = str_to_dtype(kwargs.get('dtype', 'bfloat16'))

        if source == 'local':
            if not os.path.isabs(model_id):
                model_id = os.path.join(MODELS_DIR, model_id)
            if not os.path.exists(model_id):
                raise FileNotFoundError(f"Local model {model_id} not found.")
        else:
            if not model_id.endswith('.safetensors'):
                filename = 'awq-int4-flux.1-t5xxl.safetensors'
                model_id = f"{model_id}/{filename}"

        t5 = self.graceful_model_loader(NunchakuT5EncoderModel, model_id, { "torch_dtype": dtype, "token": HF_TOKEN })

        self.mm_add(t5, priority=2)

        return { "t5": t5 }
