from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from mellon.NodeBase import NodeBase
from config import config
from utils.hf_utils import is_local_files_only
from utils.diffusers_utils import get_clip_prompt_embeds
import torch
from modules.VAE.VAE import VAEEncode
import random
import logging
logger = logging.getLogger('mellon')

HF_TOKEN = config.hf['token']

class SDXLPipelineLoader(NodeBase):
    def execute(self, model_id, dtype, variant, unet, text_encoders, vae):
        kwargs = {}

        if unet:
            kwargs['unet'] = unet

        if vae:
            kwargs['vae'] = vae

        model_id = model_id or 'stabilityai/stable-diffusion-xl-base-1.0'

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            **kwargs,
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=is_local_files_only(model_id),
            variant=variant,
            add_watermarker=False,
        )

        if text_encoders:
            pipeline.text_encoder = text_encoders['text_encoder'] if pipeline.text_encoder is not None else None
            pipeline.text_encoder_2 = text_encoders['text_encoder_2']
            pipeline.tokenizer = text_encoders['tokenizer'] if pipeline.tokenizer is not None else None
            pipeline.tokenizer_2 = text_encoders['tokenizer_2']
        
        if not hasattr(pipeline.unet, '_mm_id'):
            pipeline.unet._mm_id = self.mm_add(pipeline.unet, priority=3)

        # the refiner doesn't have the first text encoder
        if pipeline.text_encoder and not hasattr(pipeline.text_encoder, '_mm_id'):
            pipeline.text_encoder._mm_id = self.mm_add(pipeline.text_encoder, priority=1)

        if not hasattr(pipeline.text_encoder_2, '_mm_id'):
            pipeline.text_encoder_2._mm_id = self.mm_add(pipeline.text_encoder_2, priority=1)

        if not hasattr(pipeline.vae, '_mm_id'):
            pipeline.vae._mm_id = self.mm_add(pipeline.vae, priority=2)

        return {
            'pipeline': pipeline,
            'unet_out': pipeline.unet,
            'vae_out': pipeline.vae,
            'text_encoders_out': {
                'text_encoder': pipeline.text_encoder,
                'text_encoder_2': pipeline.text_encoder_2,
                'tokenizer': pipeline.tokenizer,
                'tokenizer_2': pipeline.tokenizer_2,
            },
        }

class SDXLUnetLoader(NodeBase):
    def execute(self, model_id, dtype, variant):
        model_id = model_id or 'stabilityai/stable-diffusion-xl-base-1.0'

        local_files_only = is_local_files_only(model_id)

        if not variant:
            variant = None

        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            subfolder="unet",
            token=HF_TOKEN,
            local_files_only=local_files_only,
            variant=variant,
        )

        #if not is_file_cached(model_id, 'model_index.json'):
        #    from huggingface_hub import hf_hub_download
        #    hf_hub_download(repo_id=model_id, filename='model_index.json', token=HF_TOKEN)

        unet._mm_id = self.mm_add(unet, priority=3)

        return { 'model': unet }

class SDXLTextEncodersLoader(NodeBase):
    def execute(self, model_id, dtype):
        model_id = model_id or 'stabilityai/stable-diffusion-xl-base-1.0'

        model_cfg = {
            'torch_dtype': dtype,
            'token': HF_TOKEN,
            'local_files_only': is_local_files_only(model_id),
        }

        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **model_cfg)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **model_cfg)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", **model_cfg)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", **model_cfg)
        
        text_encoder._mm_id = self.mm_add(text_encoder, priority=1)
        text_encoder_2._mm_id = self.mm_add(text_encoder_2, priority=1)

        return { 'model': {
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'text_encoder_2': text_encoder_2,
            'tokenizer_2': tokenizer_2,
        }}      


class SDXLSinglePromptEncoder(NodeBase):
    def execute(self, text_encoders, prompt, prompt_2, clip_skip, noise, prompt_scale, prompt_scale_2):
        if not isinstance(text_encoders, dict):
            text_encoders = {
                'text_encoder': text_encoders.text_encoder,
                'text_encoder_2': text_encoders.text_encoder_2,
                'tokenizer': text_encoders.tokenizer,
                'tokenizer_2': text_encoders.tokenizer_2,
            }

        prompt_embed, pooled_prompt_embed = self.encode_prompt(text_encoders, prompt, prompt_2, clip_skip, noise, prompt_scale, prompt_scale_2)

        return { 'embeds': {
            'prompt_embeds': prompt_embed,
            'pooled_prompt_embeds': pooled_prompt_embed,
        }}
    
    def encode_prompt(self, text_encoders, device, prompt="", prompt_2="", clip_skip=0, noise=0.0, prompt_scale=1.0, prompt_scale_2=1.0):
        prompt = prompt or ""
        prompt_2 = prompt_2 or prompt

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        concat_embeds = []
        if text_encoders['text_encoder']:
            text_encoders['text_encoder'] = self.mm_load(text_encoders['text_encoder'], device)
            prompt_embeds, _ = self.mm_inference(
                lambda: get_clip_prompt_embeds(prompt, text_encoders['tokenizer'], text_encoders['text_encoder'], clip_skip=clip_skip, noise=noise, scale=prompt_scale),
                device,
                exclude=text_encoders['text_encoder']
            )
            concat_embeds.append(prompt_embeds)

        text_encoders['text_encoder_2'] = self.mm_load(text_encoders['text_encoder_2'], device)
        prompt_embeds_2, pooled_prompt_embeds_2 = self.mm_inference(
            lambda: get_clip_prompt_embeds(prompt_2, text_encoders['tokenizer_2'], text_encoders['text_encoder_2'], clip_skip=clip_skip, noise=noise, scale=prompt_scale_2),
            device,
            exclude=text_encoders['text_encoder_2']
        )
        concat_embeds.append(prompt_embeds_2)

        prompt_embeds = torch.cat(concat_embeds, dim=-1).to('cpu')
        #prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1).to('cpu')
        pooled_prompt_embeds = pooled_prompt_embeds_2.to('cpu')

        return (prompt_embeds, pooled_prompt_embeds)


class SDXLPromptsEncoder(NodeBase):
    def execute(self, text_encoders, prompt, prompt_2, negative_prompt, negative_prompt_2, clip_skip, noise_positive, noise_negative, device):
        if not isinstance(text_encoders, dict):
            text_encoders = {
                'text_encoder': text_encoders.text_encoder,
                'text_encoder_2': text_encoders.text_encoder_2,
                'tokenizer': text_encoders.tokenizer,
                'tokenizer_2': text_encoders.tokenizer_2,
            }

        clip_skip = clip_skip if clip_skip > 0 else None
        prompt = prompt or ""
        prompt_2 = prompt_2 or prompt
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt

        def encode(positive_prompt, negative_prompt, text_encoder, tokenizer, clip_skip=None, noise_positive=0.0, noise_negative=0.0):
            prompt_embeds, pooled_prompt_embeds = get_clip_prompt_embeds(positive_prompt, tokenizer, text_encoder, clip_skip=clip_skip, noise=noise_positive)
            negative_prompt_embeds, negative_pooled_prompt_embeds = get_clip_prompt_embeds(negative_prompt, tokenizer, text_encoder, clip_skip=clip_skip, noise=noise_negative)
            return (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)

        # Encode the prompts with the first text encoder
        concat_embeds = []
        concat_negative_embeds = []
        if text_encoders['text_encoder']:
            text_encoders['text_encoder'] = self.mm_load(text_encoders['text_encoder'], device)
            prompt_embeds, negative_prompt_embeds, _, _ = self.mm_inference(
                lambda: encode(prompt, negative_prompt, text_encoders['text_encoder'], text_encoders['tokenizer'], clip_skip=clip_skip, noise_positive=noise_positive, noise_negative=noise_negative),
                device,
                exclude=text_encoders['text_encoder']
            )
            concat_embeds.append(prompt_embeds)
            concat_negative_embeds.append(negative_prompt_embeds)

        # Encode the prompts with the second text encoder
        text_encoders['text_encoder_2'] = self.mm_load(text_encoders['text_encoder_2'], device)
        prompt_embeds_2, negative_prompt_embeds_2, pooled_prompt_embeds_2, negative_pooled_prompt_embeds_2 = self.mm_inference(
            lambda: encode(prompt_2, negative_prompt_2, text_encoders['text_encoder_2'], text_encoders['tokenizer_2'], clip_skip=clip_skip, noise_positive=noise_positive, noise_negative=noise_negative),
            device,
            exclude=text_encoders['text_encoder_2']
        )
        concat_embeds.append(prompt_embeds_2)
        concat_negative_embeds.append(negative_prompt_embeds_2)

        # Concatenate both prompt embeddings
        prompt_embeds = torch.cat(concat_embeds, dim=-1).to('cpu')
        negative_prompt_embeds = torch.cat(concat_negative_embeds, dim=-1).to('cpu')
        pooled_prompt_embeds = pooled_prompt_embeds_2.to('cpu')
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds_2.to('cpu')
        del prompt_embeds_2, negative_prompt_embeds_2, pooled_prompt_embeds_2, negative_pooled_prompt_embeds_2, concat_embeds, concat_negative_embeds

        # Ensure both prompt embeddings have the same length
        if prompt_embeds.shape[1] > negative_prompt_embeds.shape[1]:
            negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds, (0, 0, 0, prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]))
        elif prompt_embeds.shape[1] < negative_prompt_embeds.shape[1]:
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, negative_prompt_embeds.shape[1] - prompt_embeds.shape[1]))

        return { 'embeds': {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds,
            'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
        }}

class SDXLSampler(NodeBase):
    def execute(self,
                pipeline,
                prompt,
                width,
                height,
                seed,
                steps,
                cfg,
                num_images,
                scheduler,
                latents_in,
                denoise_range,
                device,
                sync_latents,
        ):
        #generator = [torch.Generator(device=device).manual_seed(seed + i) for i in range(num_images)]
        generator = []

        random_state = random.getstate()
        random.seed(seed)
        for _ in range(num_images):
            generator.append(torch.Generator(device=device).manual_seed(seed))
            # there is a very slight chance that the seed is the same as the previous one, I don't think it's a big deal
            seed = random.randint(0, (1<<53)-1)
        random.setstate(random_state)

        denoise_range_start = denoise_range[0] if denoise_range[0] > 0 else None
        denoise_range_end = denoise_range[1] if denoise_range[1] < 1 else None

        # 1. Select the scheduler
        sampling_scheduler = pipeline.scheduler
        if ( pipeline.scheduler.__class__.__name__ != scheduler ):
            scheduler_cls = getattr(__import__('diffusers', fromlist=[scheduler]), scheduler)
            sampling_scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

        # 2. Prepare the prompts
        positive = { 'prompt_embeds': prompt['prompt_embeds'], 'pooled_prompt_embeds': prompt['pooled_prompt_embeds'] }
        negative = None

        if 'negative_prompt_embeds' in prompt:
            negative = { 'prompt_embeds': prompt['negative_prompt_embeds'], 'pooled_prompt_embeds': prompt['negative_pooled_prompt_embeds'] }

        if not negative:
            negative = { 'prompt_embeds': torch.zeros_like(positive['prompt_embeds']), 'pooled_prompt_embeds': torch.zeros_like(positive['pooled_prompt_embeds']) }
        
        # Ensure both prompt embeddings have the same length, the length might be different because our custom text encoder supports long prompts
        if positive['prompt_embeds'].shape[1] > negative['prompt_embeds'].shape[1]:
            negative['prompt_embeds'] = torch.nn.functional.pad(negative['prompt_embeds'], (0, 0, 0, positive['prompt_embeds'].shape[1] - negative['prompt_embeds'].shape[1]))
        elif positive['prompt_embeds'].shape[1] < negative['prompt_embeds'].shape[1]:
            positive['prompt_embeds'] = torch.nn.functional.pad(positive['prompt_embeds'], (0, 0, 0, negative['prompt_embeds'].shape[1] - positive['prompt_embeds'].shape[1]))

        # if the pipeline doesn't have the first text encoder, this is likely the refiner
        # so we need only the embeddings for the second text encoder
        if pipeline.text_encoder is None and positive['prompt_embeds'].shape[-1] > 1280:
            positive['prompt_embeds'] = positive['prompt_embeds'][:, :, 768:]
            negative['prompt_embeds'] = negative['prompt_embeds'][:, :, 768:]

        # 3. Latents or Images input
        denoising_start = None
        denoising_end = denoise_range_end
        image_latents = None
        strength = None
        if latents_in is not None:
            if isinstance(latents_in, torch.Tensor):
                image_latents = latents_in.clone()
                if hasattr(latents_in, '_denoising_end'):
                    denoising_start = latents_in._denoising_end
                    steps = latents_in._num_inference_steps
                else:
                    strength = 1 - (denoise_range_start or 0)
            else:
                # since we are sampling without a VAE, we need to encode the input image before passing it to the pipeline
                image_latents = VAEEncode()(model=pipeline.vae, images=latents_in, divisible_by=16, device=device)['latents']
                strength = 1 - (denoise_range_start or 0)

        if denoising_start is not None and denoising_end is not None:
            denoising_end = max(denoising_end, denoising_start + 0.01)
            logger.warning(f"Denoise range value error. Denoising end increased to: {denoising_end}")

        # 4. Run the denoise loop
        def denoise():
            # We don't need the VAE for sampling, but we need to pass something to the pipeline
            dummy_vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[128, 256, 512, 512],
                layers_per_block=2,
                latent_channels=4,
            )

            sampling_config = {
                'generator': generator,
                'prompt_embeds': positive['prompt_embeds'].to(device, dtype=pipeline.unet.dtype),
                'pooled_prompt_embeds': positive['pooled_prompt_embeds'].to(device, dtype=pipeline.unet.dtype),
                'negative_prompt_embeds': negative['prompt_embeds'].to(device, dtype=pipeline.unet.dtype),
                'negative_pooled_prompt_embeds': negative['pooled_prompt_embeds'].to(device, dtype=pipeline.unet.dtype),
                'width': width,
                'height': height,
                'guidance_scale': cfg,
                'num_inference_steps': steps,
                'output_type': "latent",
                'callback_on_step_end': self.pipe_callback,
                'denoising_start': denoising_start,
                'denoising_end': denoising_end,
                'num_images_per_prompt': num_images,
            }

            if image_latents is not None:
                sampling_config['width'] = None
                sampling_config['height'] = None
                sampling_config['image'] = image_latents.to(device)
                if strength:
                    sampling_config['strength'] = strength
                    #sampling_config['num_inference_steps'] = round(steps / strength)
                PipelineCls = StableDiffusionXLImg2ImgPipeline
                #else:
                #    sampling_config['latents'] = image_latents.to(device)
                #    PipelineCls = StableDiffusionXLPipeline
                #sampling_config['strength'] = 1 - (denoising_start or 0)
                #sampling_config['denoising_start'] = None
                #sampling_config['denoising_end'] = None
                #sampling_config['num_inference_steps'] = round(steps / sampling_config['strength'])
            else:
                PipelineCls = StableDiffusionXLPipeline

            sampling_pipe = PipelineCls.from_pretrained(
                pipeline.config._name_or_path,
                unet=pipeline.unet,
                scheduler=sampling_scheduler,
                vae=dummy_vae.to(device),
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
                local_files_only=True,
                add_watermarker=False,
            )
            sampling_pipe.watermark = None

            latents = sampling_pipe(**sampling_config).images
            del sampling_pipe, sampling_config, dummy_vae
            return latents

        self.mm_load(pipeline.unet, device)
        latents = self.mm_inference(
            denoise,
            device,
            exclude=pipeline.unet
        )
        latents = latents.to('cpu')

        if denoising_end:
            latents._denoising_end = denoising_end
            latents._num_inference_steps = steps

        return { 'latents': latents, 'pipeline_out': pipeline }
