import torch
from PIL import Image
from mellon.NodeBase import NodeBase
from utils.torch_utils import str_to_dtype, DEVICE_LIST, DEFAULT_DEVICE
from utils.memory_menager import memory_flush
from mellon.config import CONFIG
from utils.huggingface import local_files_only
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, AutoencoderKL
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from .utils import get_clip_prompt_embeds, get_t5_prompt_embeds, upcast_vae, sd3_latents_to_rgb
from utils.quantization import getQuantizationConfig, quantize

HF_TOKEN = CONFIG.hf['token']

class SD3PipelineLoader(NodeBase):
    '''description": "Load a Stable Diffusion 3 pipeline'''
    label = "SD3 Pipeline Loader"
    category = "loader"
    style = { "minWidth": 360 }
    resizable = True
    params = {
        "pipeline": { "label": "SD3 Pipeline", "display": "output", "type": ["pipeline", "StableDiffusion3Pipeline"] },
        "model_id": {
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

    def execute(self, **kwargs):
        dtype = str_to_dtype(kwargs['dtype'])
        model_id = kwargs.get('model_id', { 'source': 'hub', 'value': 'stabilityai/stable-diffusion-3.5-large'})
        source = model_id.get('source', 'hub')
        model_id = model_id.get('value', 'stabilityai/stable-diffusion-3.5-large') if isinstance(model_id, dict) else model_id

        transformer = kwargs.get('transformer', None)
        text_encoders = kwargs.get('text_encoders', None)
        load_t5 = kwargs.get('load_t5', True)
        config = {}
        
        if transformer:
            config['transformer'] = transformer
        if not load_t5:
            config['text_encoder_3'] = None
            config['tokenizer_3'] = None

        if text_encoders:
            config['text_encoder'] = text_encoders['text_encoder']
            config['text_encoder_2'] = text_encoders['text_encoder_2']
            config['text_encoder_3'] = text_encoders['text_encoder_3']
            config['tokenizer'] = text_encoders['tokenizer']
            config['tokenizer_2'] = text_encoders['tokenizer_2']
            config['tokenizer_3'] = text_encoders['tokenizer_3']

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            **config,
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=local_files_only(model_id),
        )

        if dtype == torch.float16 and pipeline.vae.config.force_upcast:
            pipeline.vae = upcast_vae(pipeline.vae)

        self.mm_add(pipeline.transformer, priority=3)
        self.mm_add(pipeline.vae, priority=2)

        #print(dict(pipeline.transformer.named_modules()).keys())

        if not text_encoders:
            self.mm_add(pipeline.text_encoder, priority=1)
            self.mm_add(pipeline.text_encoder_2, priority=1)
            if load_t5:
                self.mm_add(pipeline.text_encoder_3, priority=1)

        return { "pipeline": pipeline }

class SD3TransformerLoader(NodeBase):
    def execute(self, **kwargs):
        model_id = kwargs.get('model_id', { 'source': 'hub', 'value': 'stabilityai/stable-diffusion-3.5-large'})
        model_id = model_id.get('value', 'stabilityai/stable-diffusion-3.5-large') if isinstance(model_id, dict) else model_id

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

        loaderCallback = SD3Transformer2DModel.from_pretrained
        if quantization == 'gguf':
            from diffusers import GGUFQuantizationConfig
            config['quantization_config'] = GGUFQuantizationConfig(compute_dtype=dtype)
            loaderCallback = SD3Transformer2DModel.from_single_file
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

class SD3TextEncodersLoader(NodeBase):
    def execute(self, **kwargs):
        model_id = kwargs.get('model_id', { 'source': 'hub', 'value': 'stabilityai/stable-diffusion-3.5-large'})
        source = model_id.get('source', 'hub')
        model_id = model_id.get('value', 'stabilityai/stable-diffusion-3.5-large') if isinstance(model_id, dict) else model_id
        dtype = str_to_dtype(kwargs['dtype'])
        load_t5 = kwargs.get('load_t5', True)

        quantization = kwargs.get('quantization', 'none')
        quantization = None if quantization == 'none' else quantization

        quant_config = None
        if load_t5 and quantization == 'bnb':
            quant_config = getQuantizationConfig(quantization, **kwargs)

        config = {
            "dtype": dtype,
            "token": HF_TOKEN,
        }

        text_encoder = self.graceful_model_loader(CLIPTextModelWithProjection, model_id, {**config, "subfolder": "text_encoder"})
        tokenizer = self.graceful_model_loader(CLIPTokenizer, model_id, {**config, "subfolder": "tokenizer"})
        text_encoder_2 = self.graceful_model_loader(CLIPTextModelWithProjection, model_id, {**config, "subfolder": "text_encoder_2"})
        tokenizer_2 = self.graceful_model_loader(CLIPTokenizer, model_id, {**config, "subfolder": "tokenizer_2"})

        self.mm_add(text_encoder, priority=1)
        self.mm_add(text_encoder_2, priority=1)

        t5_encoder = None
        t5_tokenizer = None

        if load_t5:
            t5_encoder = self.graceful_model_loader(T5EncoderModel, model_id, {**config, "subfolder": "text_encoder_3", "quantization_config": quant_config})
            t5_tokenizer = self.graceful_model_loader(T5TokenizerFast, model_id, {**config, "subfolder": "tokenizer_3"})
            self.mm_add(t5_encoder, priority=0)

            #print(dict(t5_encoder.named_parameters()).keys())
            #print(dict(text_encoder_2.named_modules()).keys())

            if quantization == 'torchao' or quantization == 'quanto':
                quant_device = kwargs.get('quant_device', None)
                t5_encoder = self.mm_exec(lambda: quantize(t5_encoder, quantization, **kwargs), quant_device, exclude=[t5_encoder])
                memory_flush()
        
        return {
            "encoders": {
                "text_encoder": text_encoder,
                "text_encoder_2": text_encoder_2,
                "text_encoder_3": t5_encoder,
                "tokenizer": tokenizer,
                "tokenizer_2": tokenizer_2,
                "tokenizer_3": t5_tokenizer,
            }
        }

class SD3PromptEncoder(NodeBase):
    label = "SD3 Prompt Encoder"
    category = "embedding"
    resizable = True
    style = { "minWidth": '280px' }
    params = {
        "pipeline": { "label": "Encoders", "display": "input", "type": ["pipeline", "SD3TextEncoders"] },
        "embeds": { "label": "Embeddings", "display": "output", "type": "embedding" },
        "prompt": { "label": "Prompt", "type": "string", "display": "textarea" },
        "negative_prompt": { "label": "Negative Prompt", "type": "string", "display": "textarea" },
        "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
    }

    def execute(self, pipeline, prompt, negative_prompt, device, **kwargs):
        prompt = prompt or ''
        prompt_2 = prompt
        prompt_3 = prompt
        negative_prompt = negative_prompt or ''
        negative_prompt_2 = negative_prompt
        negative_prompt_3 = negative_prompt

        encoders = {
            'text_encoder': pipeline.text_encoder,
            'text_encoder_2': pipeline.text_encoder_2,
            'text_encoder_3': pipeline.text_encoder_3,
            'tokenizer': pipeline.tokenizer,
            'tokenizer_2': pipeline.tokenizer_2,
            'tokenizer_3': pipeline.tokenizer_3,
        }

        def encode(positive_prompt, negative_prompt, tokenizer, text_encoder):
            prompt_embeds, pooled_prompt_embeds = get_clip_prompt_embeds(positive_prompt, tokenizer, text_encoder)
            negative_prompt_embeds, negative_pooled_prompt_embeds = get_clip_prompt_embeds(negative_prompt, tokenizer, text_encoder)
            return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds
        
        # 1. encode the prompts with the first text encoder
        self.mm_load(encoders['text_encoder'], device)
        prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds = self.mm_exec(
            lambda: encode(prompt, negative_prompt, encoders['tokenizer'], encoders['text_encoder']),
            device,
            exclude=[encoders['text_encoder']],
        )

        # 2. encode the prompts with the second text encoder
        self.mm_load(encoders['text_encoder_2'], device)
        prompt_embeds_2, pooled_prompt_embeds_2, negative_prompt_embeds_2, negative_pooled_prompt_embeds_2 = self.mm_exec(
            lambda: encode(prompt_2, negative_prompt_2, encoders['tokenizer_2'], encoders['text_encoder_2']),
            device,
            exclude=[encoders['text_encoder_2']],
        )

        # 3. concatenate the prompt embeddings
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds_2], dim=-1)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1)
        negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, negative_pooled_prompt_embeds_2], dim=-1)
        
        del prompt_embeds_2, negative_prompt_embeds_2, pooled_prompt_embeds_2, negative_pooled_prompt_embeds_2

        # 4. encode the prompts with the third text encoder
        if encoders['text_encoder_3']:
            self.mm_load(encoders['text_encoder_3'], device)
            prompt_embeds_3 = self.mm_exec(
                lambda: get_t5_prompt_embeds(prompt_3, encoders['tokenizer_3'], encoders['text_encoder_3']),
                device,
                exclude=[encoders['text_encoder_3']],
            )
            negative_prompt_embeds_3 = self.mm_exec(
                lambda: get_t5_prompt_embeds(negative_prompt_3, encoders['tokenizer_3'], encoders['text_encoder_3']),
                device,
                exclude=[encoders['text_encoder_3']],
            )
        else:
            prompt_embeds_3 = torch.zeros((prompt_embeds.shape[0], 256, 4096), device='cpu', dtype=prompt_embeds.dtype)
            negative_prompt_embeds_3 = prompt_embeds_3

        del encoders
        memory_flush()

        # 5. Merge clip and T5 embedings
        # T5 should be always longer but you never know with long prompt support
        if prompt_embeds.shape[-1] > prompt_embeds_3.shape[-1]:
            prompt_embeds_3 = torch.nn.functional.pad(prompt_embeds_3, (0, prompt_embeds.shape[-1] - prompt_embeds_3.shape[-1]))
        elif prompt_embeds.shape[-1] < prompt_embeds_3.shape[-1]:
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, prompt_embeds_3.shape[-1] - prompt_embeds.shape[-1]))
        
        if negative_prompt_embeds.shape[-1] > negative_prompt_embeds_3.shape[-1]:
            negative_prompt_embeds_3 = torch.nn.functional.pad(negative_prompt_embeds_3, (0, negative_prompt_embeds.shape[-1] - negative_prompt_embeds_3.shape[-1]))
        elif negative_prompt_embeds.shape[-1] < negative_prompt_embeds_3.shape[-1]:
            negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds, (0, negative_prompt_embeds_3.shape[-1] - negative_prompt_embeds.shape[-1]))

        # concat the embedings
        prompt_embeds_3 = prompt_embeds_3
        negative_prompt_embeds_3 = negative_prompt_embeds_3
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_3], dim=-2)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds_3], dim=-2)

        del prompt_embeds_3, negative_prompt_embeds_3

        # Finally ensure positive and negative embeddings have the same length
        if prompt_embeds.shape[1] > negative_prompt_embeds.shape[1]:
            negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds, (0, 0, 0, prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]))
        elif prompt_embeds.shape[1] < negative_prompt_embeds.shape[1]:
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, negative_prompt_embeds.shape[1] - prompt_embeds.shape[1]))

        return {
            "embeds": {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            },
        }
    
class SD3LatentsPreview(NodeBase):
    label = "SD3 Latents Preview"
    category = "image"
    params = {
        "latents": { "label": "Latents", "display": "input", "type": "latent" },
        "image": { "label": "Image", "display": "output", "type": "image", "hidden": True },
        "preview": { "display": "ui_image", "dataSource": "image", "type": "url" },
    }

    def execute(self, latents, **kwargs):
        image = sd3_latents_to_rgb(latents)
        if image:
            image = image.resize((image.width * 2, image.height * 2), resample=Image.Resampling.BICUBIC)
        return { "image": image }

class SD3Sampler(NodeBase):
    label = "SD3 Sampler"
    category = "sampler"
    resizable = True
    params = {
        "pipeline": { "label": "Pipeline", "display": "input", "type": ["pipeline", "StableDiffusion3Pipeline"] },
        "embeds": { "label": "Embeddings", "display": "input", "type": "embedding" },
        "latents": { "label": "Latents", "display": "output", "type": "latent" },
        "width": { "label": "Width", "type": "int", "default": 1024, "min": 64, "step": 8 },
        "height": { "label": "Height", "type": "int", "default": 1024, "min": 64, "step": 8 },
        "seed": { "label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295 },
        "steps": { "label": "Steps", "type": "int", "display": "slider", "default": 30, "min": 1, "max": 100 },
        "cfg": { "label": "Guidance", "type": "float", "display": "slider","default": 5, "min": 0.0, "max": 50.0, "step": 0.5 },
        "cfg_cutoff": { "label": "Enable CFG Cutoff", "type": "boolean", "default": False, "onChange": { True: ["cfg_step"], False: [] } },
        "cfg_step": { "label": "Cutoff Step", "type": "float", "display": "slider", "default": 0.5, "min": 0, "max": 1, "step": 0.01, "onChange": { True: ["cfg_cutoff"], False: [] } },
        "scheduler": { "label": "Scheduler", "type": "string", "options": {
            "FlowMatchEulerDiscreteScheduler": "Flow Match Euler Discrete",
            "FlowMatchHeunDiscreteScheduler": "Flow Match Heun Discrete",
        }, "default": "FlowMatchEulerDiscreteScheduler" },
        "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
        "latents_preview": { "label": "Latents Preview", "display": "output", "type": "latent" },
    }

    def execute(self, pipeline, **kwargs):
        embeds = kwargs['embeds']
        width = kwargs.get('width', 1024)
        height = kwargs.get('height', 1024)
        seed = kwargs.get('seed', 0)
        steps = kwargs.get('steps', 30)
        cfg = kwargs.get('cfg', 5)
        cfg_cutoff = kwargs.get('cfg_cutoff', False)
        cfg_step = kwargs.get('cfg_step', 0)
        scheduler = kwargs.get('scheduler', 'FlowMatchEulerDiscreteScheduler')
        device = kwargs.get('device', DEFAULT_DEVICE)

        generator = torch.Generator(device=device).manual_seed(seed)

        # 1. Create the scheduler
        use_dynamic_shifting = True
        if ( pipeline.scheduler.__class__.__name__ != scheduler ):
            if scheduler == 'FlowMatchHeunDiscreteScheduler':
                from diffusers import FlowMatchHeunDiscreteScheduler as SchedulerCls
                use_dynamic_shifting = False # not supported by Heun
            else:
                from diffusers import FlowMatchEulerDiscreteScheduler as SchedulerCls
        else:
            SchedulerCls = pipeline.scheduler.__class__

        scheduler_config = pipeline.scheduler.config
        sampling_scheduler = SchedulerCls.from_config(scheduler_config, use_dynamic_shifting=use_dynamic_shifting)

        # 2. Prepare the prompts
        positive = { "prompt_embeds": embeds['prompt_embeds'], "pooled_prompt_embeds": embeds['pooled_prompt_embeds'] }
        negative = None

        if 'negative_prompt_embeds' in embeds:
            negative = { "prompt_embeds": embeds['negative_prompt_embeds'], "pooled_prompt_embeds": embeds['negative_pooled_prompt_embeds'] }

        if not negative:
            negative = { 'prompt_embeds': torch.zeros_like(positive['prompt_embeds']), 'pooled_prompt_embeds': torch.zeros_like(positive['pooled_prompt_embeds']) }

        # Ensure both prompt embeddings have the same length
        if positive['prompt_embeds'].shape[1] > negative['prompt_embeds'].shape[1]:
            negative['prompt_embeds'] = torch.nn.functional.pad(negative['prompt_embeds'], (0, 0, 0, positive['prompt_embeds'].shape[1] - negative['prompt_embeds'].shape[1]))
        elif positive['prompt_embeds'].shape[1] < negative['prompt_embeds'].shape[1]:
            positive['prompt_embeds'] = torch.nn.functional.pad(positive['prompt_embeds'], (0, 0, 0, negative['prompt_embeds'].shape[1] - positive['prompt_embeds'].shape[1]))

        dummy_vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            latent_channels=16,
        )

        sampling_pipeline = StableDiffusion3Pipeline.from_pretrained(
            pipeline.config._name_or_path,
            transformer=pipeline.transformer,
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
            scheduler=sampling_scheduler,
            local_files_only=True,
            vae=dummy_vae,
        )

        def preview_callback(pipe, step_index, timestep, callback_kwargs):           
            latents = callback_kwargs['latents']
            self.trigger_output("latents_preview", latents)       
            self.pipe_callback(pipe, step_index, timestep, callback_kwargs)
            return callback_kwargs

        sampling_config = {
            'generator': generator,
            'prompt_embeds': positive['prompt_embeds'],
            'pooled_prompt_embeds': positive['pooled_prompt_embeds'],
            'negative_prompt_embeds': negative['prompt_embeds'],
            'negative_pooled_prompt_embeds': negative['pooled_prompt_embeds'],
            'width': width,
            'height': height,
            'guidance_scale': cfg,
            'num_inference_steps': steps,
            'output_type': "latent",
            'callback_on_step_end': preview_callback,
            #'num_images_per_prompt': 1, TODO: add support for multiple images
        }

        if cfg_cutoff:
            sampling_pipeline._cfg_cutoff_step = cfg_step
            sampling_config['callback_on_step_end_tensor_inputs'] = ["latents", "prompt_embeds", "pooled_prompt_embeds"]

        del positive, negative, pipeline, embeds
    
        # 3. Run the denoise loop
        def sampling(pipe, config, device):
            pipe.vae.to(device)
            config['prompt_embeds'] = config['prompt_embeds'].to(device, dtype=pipe.transformer.dtype)
            config['pooled_prompt_embeds'] = config['pooled_prompt_embeds'].to(device, dtype=pipe.transformer.dtype)
            config['negative_prompt_embeds'] = config['negative_prompt_embeds'].to(device, dtype=pipe.transformer.dtype)
            config['negative_pooled_prompt_embeds'] = config['negative_pooled_prompt_embeds'].to(device, dtype=pipe.transformer.dtype)

            latents = pipe(**config).images
            config['prompt_embeds'] = config['prompt_embeds'].to('cpu')
            config['pooled_prompt_embeds'] = config['pooled_prompt_embeds'].to('cpu')
            config['negative_prompt_embeds'] = config['negative_prompt_embeds'].to('cpu')
            config['negative_pooled_prompt_embeds'] = config['negative_pooled_prompt_embeds'].to('cpu')
            del pipe, config
            return latents.to('cpu').detach().clone()

        self.mm_load(sampling_pipeline.transformer, device)
        latents = self.mm_exec(
            lambda: sampling(sampling_pipeline, sampling_config, device),
            device,
            exclude=[sampling_pipeline.transformer],
        )

        del sampling_pipeline, sampling_config, dummy_vae

        return { "latents": latents, "latents_preview": latents }
