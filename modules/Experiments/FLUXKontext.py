from mellon.NodeBase import NodeBase
from utils.torch_utils import str_to_dtype, DEVICE_LIST, DEFAULT_DEVICE
from utils.huggingface import local_files_only, get_local_model_ids
from huggingface_hub import snapshot_download
from mellon.config import CONFIG
from diffusers import FluxKontextPipeline, BitsAndBytesConfig, FluxTransformer2DModel, AutoencoderKL
from modules.Experiments import QUANT_FIELDS, QUANT_SELECT, PREFERRED_KONTEXT_RESOLUTIONS
from utils.quantization import getBnBConfig
from utils.image import fit as image_fit
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from utils.memory_menager import memory_flush
import torch

HF_TOKEN = CONFIG.hf['token']

class FluxTransformerLoader(NodeBase):
    label = "FLUX Transformer Loader"
    category = "loader"
    params = {
        "model_id": {
            "label": "Model",
            "display": "autocomplete",
            "type": "string",
            "default": "black-forest-labs/FLUX.1-Kontext-dev",
            "optionsSource": { "source": "hf_cache", "filter": { "className": "FluxKontextPipeline" } },
            "fieldOptions": { "noValidation": True, "model_loader": True }
        },
        "dtype": {
            "label": "Dtype",
            "type": "string",
            "default": "bfloat16",
            "options": ['auto', 'float32', 'float16', 'bfloat16'],
        },
        **QUANT_SELECT,
        **QUANT_FIELDS,
        "transformer": { "label": "Transformer", "display": "output", "type": "FluxTransformer2DModel" },
    }

    def execute(self, **kwargs):
        model_id = kwargs.get('model_id', 'black-forest-labs/FLUX.1-Kontext-dev')
        dtype = str_to_dtype(kwargs['dtype'])
        quantization = kwargs.get('quantization', 'none')
        bnb_type = kwargs.get('bnb_type', '8bit')
        bnb_double_quant = kwargs.get('bnb_double_quant', True)

        quant_config = {}
        if quantization == 'bnb':
            quant_config = getBnBConfig(bnb_type, dtype, bnb_double_quant)

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=True,
            quantization_config=quant_config,
        )

        self.mm_add(transformer, priority=3)

        return { "transformer": transformer }

class FluxTextEncoderLoader(NodeBase):
    label = "FLUX Text Encoder Loader"
    category = "loader"
    resizeable = True
    params = {
        "model_id": {
            "label": "Model",
            "display": "autocomplete",
            "type": "string",
            "default": "black-forest-labs/FLUX.1-Kontext-dev",
            "optionsSource": { "source": "hf_cache", "filter": { "className": "FluxKontextPipeline" } },
            "fieldOptions": { "noValidation": True, "model_loader": True }
        },
        "dtype": {
            "label": "Dtype",
            "type": "string",
            "default": "bfloat16",
            "options": ['auto', 'float32', 'float16', 'bfloat16'],
        },
        **QUANT_SELECT,
        **QUANT_FIELDS,
        "encoders": {
            "label": "Encoders",
            "display": "output",
            "type": "FluxTextEncoders",
        },
    }
    def execute(self, **kwargs):
        model_id = kwargs.get('model_id', 'black-forest-labs/FLUX.1-Kontext-dev')
        dtype = str_to_dtype(kwargs.get('dtype', 'bfloat16'))
        quantization = kwargs.get('quantization', 'none')
        bnb_type = kwargs.get('bnb_type', '8bit')
        bnb_double_quant = kwargs.get('bnb_double_quant', True)

        quant_config = {}
        if quantization == 'bnb':
            quant_config = getBnBConfig(bnb_type, dtype, bnb_double_quant)

        config = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only(model_id),
            "token": HF_TOKEN,
        }

        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **config)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **config)
        text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", **config, quantization_config=quant_config)
        tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2", **config)

        self.mm_add(text_encoder, priority=1)
        self.mm_add(text_encoder_2, priority=1)

        return { "encoders": { "text_encoder": text_encoder, "text_encoder_2": text_encoder_2, "tokenizer": tokenizer, "tokenizer_2": tokenizer_2 } }

class FluxPipelineLoader(NodeBase):
    label = "FLUX Kontext Pipeline Loader"
    category = "loader"
    params = {
        "pipeline": { "label": "FLUX Pipeline", "display": "output", "type": "pipeline" },
        "model_id": {
            "label": "Model",
            "display": "autocomplete",
            "type": "string",
            "default": "black-forest-labs/FLUX.1-Kontext-dev",
            "optionsSource": { "source": "hf_cache", "filter": { "className": "FluxKontextPipeline" } },
            "fieldOptions": { "noValidation": True, "model_loader": True }
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
        dtype = str_to_dtype(kwargs['dtype'])
        model_id = kwargs.get('model_id', 'black-forest-labs/FLUX.1-Kontext-dev')
        transformer = kwargs.get('transformer', None)
        encoders = kwargs.get('encoders', None)
        if not local_files_only(model_id):
            # ignore the merge file
            snapshot_download(repo_id=model_id, token=HF_TOKEN, ignore_patterns=["flux1-kontext-dev.safetensors"])

        config = {}
        if transformer:
            config['transformer'] = transformer
        if encoders:
            config['text_encoder'] = encoders['text_encoder']
            config['text_encoder_2'] = encoders['text_encoder_2']
            config['tokenizer'] = encoders['tokenizer']
            config['tokenizer_2'] = encoders['tokenizer_2']

        pipeline = FluxKontextPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=True,
            **config,
        )

        self.mm_add(pipeline.vae, priority=2)
        if not encoders:
            self.mm_add(pipeline.text_encoder, priority=1)
            self.mm_add(pipeline.text_encoder_2, priority=1)

        if not transformer:
            self.mm_add(pipeline.transformer, priority=3)

        return { "pipeline": pipeline }

class FluxKontextTextEncoder(NodeBase):
    label = "FLUX Kontext Text Encoder"
    category = "embedding"
    params = {
        "pipeline": { "label": "FLUX Pipeline", "display": "input", "type": ["pipeline", "FluxTextEncoders"] },
        "embeds": { "label": "Embeddings", "display": "output", "type": "embedding" },
        "prompt": { "label": "Prompt", "type": "text" },
        #"negative_prompt": { "label": "Negative Prompt", "type": "text" },
        "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
    }

    def execute(self, **kwargs):
        pipeline = kwargs['pipeline']
        prompt = kwargs.get('prompt', '')
        device = kwargs.get('device', DEFAULT_DEVICE)

        work_pipe = FluxKontextPipeline.from_pretrained(
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
    def execute(self, **kwargs):
        pipeline = kwargs['pipeline']
        prompt_embeds, pooled_prompt_embeds = kwargs.get('embeds', (None, None))
        cfg = kwargs['cfg']
        image = kwargs['image']
        resolution = kwargs.get('resolution', '1024x1024')
        width, height = [int(x) for x in resolution.split('x')]
        device = kwargs.get('device', DEFAULT_DEVICE)
        seed = kwargs.get('seed', 0)
        steps = kwargs.get('steps', 25)
        
        generator = torch.Generator(device=device).manual_seed(seed)

        encode_pipe = FluxKontextPipeline.from_pretrained(
            pipeline.config._name_or_path,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            transformer=None,
            vae=pipeline.vae,
            local_files_only=True,
        )

        def encode_image(pipe, image, width, height, device, generator):
            dtype = pipe.vae.dtype
            image = image_fit(image, width, height)
            image = pipe.image_processor.preprocess(image)
            image = image.to(device, dtype=dtype)
            image_latents = pipe._encode_vae_image(image, generator)
            image_latents = image_latents.to('cpu').detach().clone()
            del pipe, image
            return image_latents

        image_latents = self.mm_exec(
            lambda: encode_image(encode_pipe, image, width, height, device, generator),
            device,
            models=[encode_pipe.vae],
        )
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

        sampling_pipeline = FluxKontextPipeline.from_pretrained(
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
        
        print("Sampling")

        latents = self.mm_exec(
            lambda: sampling(sampling_pipeline, sampling_config, device),
            device,
            models=[sampling_pipeline.transformer],
        )
        del sampling_pipeline, dummy_vae, image_latents, prompt_embeds, pooled_prompt_embeds

        #latents = latents.to('cpu').detach().clone()

        return { "latents": (latents, (height, width)) }