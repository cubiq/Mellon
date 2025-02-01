import logging
logger = logging.getLogger('mellon')
import torch
from diffusers import SD3Transformer2DModel, AutoencoderKL, StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from mellon.NodeBase import NodeBase
from utils.hf_utils import is_local_files_only, get_repo_path
from utils.diffusers_utils import get_clip_prompt_embeds, get_t5_prompt_embeds
from config import config
from mellon.quantization import NodeQuantization
import math

HF_TOKEN = config.hf['token']

def calculate_mu(width: int, height: int, 
                patch_size: int = 2,
                base_image_seq_len: int = 256,
                max_image_seq_len: int = 4096,
                base_shift: float = 0.5,
                max_shift: float = 1.15) -> float:

    # latent size
    width = width // 8
    height = height // 8

    seq_len = (width // patch_size) * (height // patch_size)
    seq_len = max(min(seq_len, max_image_seq_len), base_image_seq_len)

    # this is the default mu calculation
    #m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    #b = base_shift - m * base_image_seq_len
    #mu = seq_len * m + b
    
    # This is my own mess. TODO: check if this is correct
    factor = (math.log2(seq_len) - math.log2(base_image_seq_len)) / (math.log2(max_image_seq_len) - math.log2(base_image_seq_len))
    factor = max(min(factor, 1.0), 0.0)
    mu = base_shift + factor * (max_shift - base_shift)

    return mu

class SD3PipelineLoader(NodeBase):
    def execute(
            self,
            model_id,
            dtype,
            load_t5,
            transformer_in,
            text_encoders_in,
            vae_in,
        ):
        kwargs = {}

        if transformer_in:
            kwargs['transformer'] = transformer_in

        if text_encoders_in:
            kwargs['text_encoder'] = text_encoders_in['text_encoder']
            kwargs['text_encoder_2'] = text_encoders_in['text_encoder_2']
            kwargs['text_encoder_3'] = text_encoders_in['text_encoder_3']
            kwargs['tokenizer'] = text_encoders_in['tokenizer']
            kwargs['tokenizer_2'] = text_encoders_in['tokenizer_2']
            kwargs['tokenizer_3'] = text_encoders_in['tokenizer_3']

        if vae_in:
            kwargs['vae'] = vae_in

        if not load_t5:
            kwargs['text_encoder_3'] = None
            kwargs['tokenizer_3'] = None

        model_id = model_id or 'stabilityai/stable-diffusion-3.5-large'

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            **kwargs,
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=is_local_files_only(model_id),
        )

        if not hasattr(pipeline.transformer, '_mm_id'):
            pipeline.transformer._mm_id = self.mm_add(pipeline.transformer, priority=3)

        if not hasattr(pipeline.text_encoder, '_mm_id'):
            pipeline.text_encoder._mm_id = self.mm_add(pipeline.text_encoder, priority=1)

        if not hasattr(pipeline.text_encoder_2, '_mm_id'):
            pipeline.text_encoder_2._mm_id = self.mm_add(pipeline.text_encoder_2, priority=1)

        if load_t5 and not hasattr(pipeline.text_encoder_3, '_mm_id'):
            pipeline.text_encoder_3._mm_id = self.mm_add(pipeline.text_encoder_3, priority=1)

        if not hasattr(pipeline.vae, '_mm_id'):
            pipeline.vae._mm_id = self.mm_add(pipeline.vae, priority=2)

        return {
            'pipeline': pipeline,
        }

class SD3TransformerLoader(NodeBase, NodeQuantization):
    def execute(self, model_id, dtype, compile, quantization, **kwargs):
        import os
        model_id = model_id or 'stabilityai/stable-diffusion-3.5-large'

        local_files_only = is_local_files_only(model_id)

        # overcome bug in diffusers loader with sharded weights
        if local_files_only:
            model_path = os.path.join(get_repo_path(model_id), "transformer")
        else:
            model_path = model_id

        quantization_config = None
        if quantization == 'bitsandbytes':
            from mellon.quantization import bitsandbytes
            quantization_config = bitsandbytes(kwargs['bitsandbytes_weights'], dtype=dtype, double_quant=kwargs['bitsandbytes_double_quant'])

        transformer_model = SD3Transformer2DModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            subfolder="transformer" if not local_files_only else None,
            token=HF_TOKEN,
            local_files_only=local_files_only,
            quantization_config=quantization_config,
        )

        transformer_model._mm_id = self.mm_add(transformer_model, priority=3)

        if quantization != 'none' and not quantization_config:
            transformer_model = self.quantize(quantization, model=transformer_model._mm_id, **kwargs)

        if compile:
            from utils.memory_manager import memory_manager
            from utils.torch_utils import compile
            memory_manager.unload_all(exclude=transformer_model._mm_id)
            transformer_model = compile(transformer_model)
            self.mm_update(transformer_model._mm_id, model=transformer_model)

        return { 'model': transformer_model }


class SD3TextEncodersLoader(NodeBase, NodeQuantization):
    def execute(self, model_id, dtype, load_t5, quantization, **kwargs):
        model_id = model_id or 'stabilityai/stable-diffusion-3.5-large'

        model_cfg = {
            'torch_dtype': dtype,
            'token': HF_TOKEN,
            'local_files_only': is_local_files_only(model_id),
            #'use_safetensors': True,
        }

        text_encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder", **model_cfg)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **model_cfg)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", **model_cfg)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", **model_cfg)

        text_encoder._mm_id = self.mm_add(text_encoder, priority=1)
        text_encoder_2._mm_id = self.mm_add(text_encoder_2, priority=1)

        t5_encoder = None
        t5_tokenizer = None
        if load_t5:
            if quantization == 'bitsandbytes':
                from mellon.quantization import bitsandbytes
                model_cfg['quantization_config'] = bitsandbytes(kwargs['bitsandbytes_weights'], dtype=dtype, double_quant=kwargs['bitsandbytes_double_quant'])

            t5_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3", **model_cfg)
            t5_tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3", **model_cfg)
            t5_encoder._mm_id = self.mm_add(t5_encoder, priority=0)

            if quantization != 'none' and not 'quantization_config' in model_cfg:
                self.quantize(quantization, model=t5_encoder._mm_id, **kwargs)

        return { 'model': {
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'text_encoder_2': text_encoder_2,
            'tokenizer_2': tokenizer_2,
            'text_encoder_3': t5_encoder,
            'tokenizer_3': t5_tokenizer,
        }}

class SD3PromptEncoder(NodeBase):
    def execute(self,
                text_encoders,
                prompt,
                prompt_2,
                prompt_3,
                negative_prompt,
                negative_prompt_2,
                negative_prompt_3,
                noise_clip,
                noise_negative_clip,
                noise_t5,
                noise_negative_t5,
                device):
        
        if not isinstance(text_encoders, dict):
            text_encoders = {
                'text_encoder': text_encoders.text_encoder,
                'text_encoder_2': text_encoders.text_encoder_2,
                'text_encoder_3': text_encoders.text_encoder_3,
                'tokenizer': text_encoders.tokenizer,
                'tokenizer_2': text_encoders.tokenizer_2,
                'tokenizer_3': text_encoders.tokenizer_3,
            }

        prompt = prompt or ""
        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt
        negative_prompt_3 = negative_prompt_3 or negative_prompt

        def encode(positive_prompt, negative_prompt, text_encoder, tokenizer, clip_skip=None, noise=0.0, negative_noise=0.0):
            prompt_embeds, pooled_prompt_embeds = get_clip_prompt_embeds(positive_prompt, tokenizer, text_encoder, clip_skip=clip_skip, noise=noise)
            negative_prompt_embeds, negative_pooled_prompt_embeds = get_clip_prompt_embeds(negative_prompt, tokenizer, text_encoder, clip_skip=clip_skip, noise=negative_noise)
            return (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)

        # 1. Encode the prompts with the first text encoder
        text_encoders['text_encoder'] = self.mm_load(text_encoders['text_encoder'], device)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.mm_inference(
            lambda: encode(prompt, negative_prompt, text_encoders['text_encoder'], text_encoders['tokenizer'], noise=noise_clip, negative_noise=noise_negative_clip),
            device,
            exclude=text_encoders['text_encoder']
        )

        # 2. Encode the prompts with the second text encoder
        text_encoders['text_encoder_2'] = self.mm_load(text_encoders['text_encoder_2'], device)
        prompt_embeds_2, negative_prompt_embeds_2, pooled_prompt_embeds_2, negative_pooled_prompt_embeds_2 = self.mm_inference(
            lambda: encode(prompt_2, negative_prompt_2, text_encoders['text_encoder_2'], text_encoders['tokenizer_2'], noise=noise_clip, negative_noise=noise_negative_clip),
            device,
            exclude=text_encoders['text_encoder_2']
        )

        # 3. Concatenate all clip embeddings
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1).to('cpu')
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds_2], dim=-1).to('cpu')
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1).to('cpu')
        negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, negative_pooled_prompt_embeds_2], dim=-1).to('cpu')
        del prompt_embeds_2, negative_prompt_embeds_2, pooled_prompt_embeds_2, negative_pooled_prompt_embeds_2

        # 4. Encode the prompts with the third text encoder
        if text_encoders['text_encoder_3']:
            text_encoders['text_encoder_3'] = self.mm_load(text_encoders['text_encoder_3'], device)
            prompt_embeds_3 = self.mm_inference(
                lambda: get_t5_prompt_embeds(prompt_3, text_encoders['tokenizer_3'], text_encoders['text_encoder_3'], noise=noise_t5),
                device,
                exclude=text_encoders['text_encoder_3']
            )
            negative_prompt_embeds_3 = self.mm_inference(   
                lambda: get_t5_prompt_embeds(negative_prompt_3, text_encoders['tokenizer_3'], text_encoders['text_encoder_3'], noise=noise_negative_t5),
                device,
                exclude=text_encoders['text_encoder_3']
            )
        else:
            prompt_embeds_3 = torch.zeros((prompt_embeds.shape[0], 256, 4096), device='cpu', dtype=prompt_embeds.dtype)
            negative_prompt_embeds_3 = prompt_embeds_3

        # 5. Merge the clip and T5 embedings
        # T5 should be always longer but you never know with long prompt support
        if prompt_embeds.shape[-1] > prompt_embeds_3.shape[-1]:
            prompt_embeds_3 = torch.nn.functional.pad(prompt_embeds_3, (0, prompt_embeds.shape[-1] - prompt_embeds_3.shape[-1]))
        elif prompt_embeds.shape[-1] < prompt_embeds_3.shape[-1]:
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, prompt_embeds_3.shape[-1] - prompt_embeds.shape[-1]))
        if negative_prompt_embeds.shape[-1] > negative_prompt_embeds_3.shape[-1]:
            negative_prompt_embeds_3 = torch.nn.functional.pad(negative_prompt_embeds_3, (0, negative_prompt_embeds.shape[-1] - negative_prompt_embeds_3.shape[-1]))
        elif negative_prompt_embeds.shape[-1] < negative_prompt_embeds_3.shape[-1]:
            negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds, (0, negative_prompt_embeds_3.shape[-1] - negative_prompt_embeds.shape[-1]))

        prompt_embeds_3 = prompt_embeds_3.to('cpu')
        negative_prompt_embeds_3 = negative_prompt_embeds_3.to('cpu')
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_3], dim=-2)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds_3], dim=-2)

        # Finally ensure positive and negative prompt embeddings have the same length
        if prompt_embeds.shape[1] > negative_prompt_embeds.shape[1]:
            negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds, (0, 0, 0, prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]))
        elif prompt_embeds.shape[1] < negative_prompt_embeds.shape[1]:
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, negative_prompt_embeds.shape[1] - prompt_embeds.shape[1]))

        return {
            'embeds': {
                'prompt_embeds': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds,
                'negative_prompt_embeds': negative_prompt_embeds,
                'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
            }
        }

class SD3Sampler(NodeBase):
    # def __init__(self, node_id):
    #     super().__init__(node_id)

    #     self.dummy_vae = AutoencoderKL(
    #         in_channels=3,
    #         out_channels=3,
    #         down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    #         up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    #         block_out_channels=[128, 256, 512, 512],
    #         layers_per_block=2,
    #         latent_channels=16,
    #     )

    #     self.schedulers_config = {
    #         'FlowMatchEulerDiscreteScheduler': {
    #             'num_train_timesteps': 1000,
    #             'shift': 3.0,
    #             'use_dynamic_shifting': False,
    #             'base_shift': 0.5,
    #             'max_shift': 1.15,
    #             'base_image_seq_len': 256,
    #             'max_image_seq_len': 4096,
    #             'invert_sigmas': False,
    #         },
    #         'FlowMatchHeunDiscreteScheduler': {
    #             'num_train_timesteps': 1000,
    #             'shift': 3.0,
    #         }
    #     }
        
    def execute(self,
                pipeline,
                prompt,
                width,
                height,
                seed,
                latents_in,
                scheduler,
                steps,
                cfg,
                denoise_range,
                shift,
                use_dynamic_shifting,
                device):

        generator = torch.Generator(device=device).manual_seed(seed)

        # 1. Create the scheduler
        if ( pipeline.scheduler.__class__.__name__ != scheduler ):
            if scheduler == 'FlowMatchHeunDiscreteScheduler':
                from diffusers import FlowMatchHeunDiscreteScheduler as SchedulerCls
                use_dynamic_shifting = False # not supported by Heun
            else:
                from diffusers import FlowMatchEulerDiscreteScheduler as SchedulerCls
        else:
            SchedulerCls = pipeline.scheduler.__class__

        scheduler_config = pipeline.scheduler.config
        mu = None
        if use_dynamic_shifting:
            mu = calculate_mu(width, height,
                            patch_size=2,
                            base_image_seq_len=scheduler_config['base_image_seq_len'],
                            max_image_seq_len=scheduler_config['max_image_seq_len'],
                            base_shift=scheduler_config['base_shift'],
                            max_shift=scheduler_config['max_shift'])

        sampling_scheduler = SchedulerCls.from_config(scheduler_config, shift=shift, use_dynamic_shifting=use_dynamic_shifting)

        # 2. Prepare the prompts
        positive = { 'prompt_embeds': prompt['prompt_embeds'], 'pooled_prompt_embeds': prompt['pooled_prompt_embeds'] }
        negative = None

        if 'negative_prompt_embeds' in prompt:
            negative = { 'prompt_embeds': prompt['negative_prompt_embeds'], 'pooled_prompt_embeds': prompt['negative_pooled_prompt_embeds'] }

        if not negative:
            negative = { 'prompt_embeds': torch.zeros_like(positive['prompt_embeds']), 'pooled_prompt_embeds': torch.zeros_like(positive['pooled_prompt_embeds']) }
        
        # Ensure both prompt embeddings have the same length
        if positive['prompt_embeds'].shape[1] > negative['prompt_embeds'].shape[1]:
            negative['prompt_embeds'] = torch.nn.functional.pad(negative['prompt_embeds'], (0, 0, 0, positive['prompt_embeds'].shape[1] - negative['prompt_embeds'].shape[1]))
        elif positive['prompt_embeds'].shape[1] < negative['prompt_embeds'].shape[1]:
            positive['prompt_embeds'] = torch.nn.functional.pad(positive['prompt_embeds'], (0, 0, 0, negative['prompt_embeds'].shape[1] - positive['prompt_embeds'].shape[1]))

        # 3. Run the denoise loop
        pipelineCls = StableDiffusion3Pipeline if latents_in is None else StableDiffusion3Img2ImgPipeline

        def sampling():
            dummy_vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
                up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
                block_out_channels=[128, 256, 512, 512],
                layers_per_block=2,
                latent_channels=16,
            )

            sampling_pipe = pipelineCls.from_pretrained(
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
                vae=dummy_vae.to(device),
            )

            sampling_config = {
                'generator': generator,
                'prompt_embeds': positive['prompt_embeds'].to(device, dtype=pipeline.transformer.dtype),
                'pooled_prompt_embeds': positive['pooled_prompt_embeds'].to(device, dtype=pipeline.transformer.dtype),
                'negative_prompt_embeds': negative['prompt_embeds'].to(device, dtype=pipeline.transformer.dtype),
                'negative_pooled_prompt_embeds': negative['pooled_prompt_embeds'].to(device, dtype=pipeline.transformer.dtype),
                'width': width,
                'height': height,
                'guidance_scale': cfg,
                'num_inference_steps': steps,
                'output_type': "latent",
                'callback_on_step_end': self.pipe_callback,
                'mu': mu,
            }

            if latents_in is not None:
                sampling_config['width'] = None
                sampling_config['height'] = None
                sampling_config['image'] = latents_in
                sampling_config['strength'] = 1 - (denoise_range[0] or 0)

            latents = sampling_pipe(**sampling_config).images
            del sampling_pipe, sampling_config, dummy_vae
            return latents

        self.mm_load(pipeline.transformer, device)
        latents = self.mm_inference(
            sampling,
            device,
            exclude=pipeline.transformer
        )
        latents = latents.to('cpu')

        return { 'latents': latents, 'pipeline_out': pipeline }
