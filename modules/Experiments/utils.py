import torch
from PIL import Image

def get_clip_prompt_embeds(prompt, tokenizer, text_encoder, clip_skip=None, noise=0.0, scale=1.0):
    max_length = tokenizer.model_max_length
    device = text_encoder.device
    bos = torch.tensor([tokenizer.bos_token_id], device=device).unsqueeze(0)
    eos = torch.tensor([tokenizer.eos_token_id], device=device).unsqueeze(0)
    one = torch.tensor([1], device=device).unsqueeze(0)
    pad = tokenizer.pad_token_id

    text_input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(device)

    # remove start and end tokens
    text_input_ids = text_input_ids[:, 1:-1]

    # we create chunks of max_length-2, we add start and end tokens back later
    chunks = text_input_ids.split(max_length-2, dim=-1)

    concat_embeds = []
    pooled_prompt_embeds = None
    for chunk in chunks:
        mask = torch.ones_like(chunk)

        # add start and end tokens to each chunk
        chunk = torch.cat([bos, chunk, eos], dim=-1)
        mask = torch.cat([one, mask, one], dim=-1)

        # pad the chunk to the max length
        if chunk.shape[-1] < max_length:
            mask = torch.nn.functional.pad(mask, (0, max_length - mask.shape[-1]), value=0)
            chunk = torch.nn.functional.pad(chunk, (0, max_length - chunk.shape[-1]), value=pad)

        # encode the tokenized text
        prompt_embeds = text_encoder(chunk, attention_mask=mask, output_hidden_states=True)
        
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        concat_embeds.append(prompt_embeds)

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    del text_encoder, bos, eos, one, pad, text_input_ids, chunks, concat_embeds, mask, chunk

    if scale != 1.0:
        prompt_embeds = prompt_embeds * scale
        pooled_prompt_embeds = pooled_prompt_embeds * scale

    if noise > 0.0:
        generator_state = torch.get_rng_state()

        seed = int(prompt_embeds.mean().item() * 1e6) % (2**32 - 1)
        torch.manual_seed(seed)
        embed_noise = torch.randn_like(prompt_embeds) * prompt_embeds.abs().mean() * noise
        #embed_noise = torch.randn_like(prompt_embeds) * noise
        prompt_embeds = prompt_embeds + embed_noise

        seed = int(pooled_prompt_embeds.mean().item() * 1e6) % (2**32 - 1)
        torch.manual_seed(seed)
        embed_noise = torch.randn_like(pooled_prompt_embeds) * pooled_prompt_embeds.abs().mean() * noise
        #embed_noise = torch.randn_like(pooled_prompt_embeds) * noise
        pooled_prompt_embeds = pooled_prompt_embeds + embed_noise

        torch.set_rng_state(generator_state)
    
    prompt_embeds = prompt_embeds.to('cpu').detach().clone()
    pooled_prompt_embeds = pooled_prompt_embeds.to('cpu').detach().clone()

    return prompt_embeds, pooled_prompt_embeds


def get_t5_prompt_embeds(prompt, tokenizer, text_encoder, max_sequence_length=256, noise=0.0):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # could be tokenizer.model_max_length but we are using a more conservative value (256)
    max_length = max_sequence_length
    device = text_encoder.device
    eos = torch.tensor([1], device=device).unsqueeze(0)
    pad = 0 # pad token is 0

    text_inputs_ids = tokenizer(prompt, truncation = False, add_special_tokens=True, return_tensors="pt").input_ids.to(device)

    # remove end token
    text_inputs_ids = text_inputs_ids[:, :-1]

    chunks = text_inputs_ids.split(max_length-1, dim=-1)

    concat_embeds = []
    for chunk in chunks:
        mask = torch.ones_like(chunk)

        # add end token back
        chunk = torch.cat([chunk, eos], dim=-1)
        mask = torch.cat([mask, eos], dim=-1)

        # pad the chunk to the max length
        if chunk.shape[-1] < max_length:
            mask = torch.nn.functional.pad(mask, (0, max_length - mask.shape[-1]), value=0)
            chunk = torch.nn.functional.pad(chunk, (0, max_length - chunk.shape[-1]), value=pad)

        # encode the tokenized text
        prompt_embeds = text_encoder(chunk)[0]
        concat_embeds.append(prompt_embeds)

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    del text_encoder, eos, pad, text_inputs_ids, chunks, concat_embeds, mask, chunk

    if noise > 0.0:
        generator_state = torch.get_rng_state()
        seed = int(prompt_embeds.mean().item() * 1e6) % (2**32 - 1)
        torch.manual_seed(seed)
        embed_noise = torch.randn_like(prompt_embeds) * prompt_embeds.abs().mean() * noise
        prompt_embeds = prompt_embeds + embed_noise
        torch.set_rng_state(generator_state)

    prompt_embeds = prompt_embeds.to('cpu').detach().clone()

    return prompt_embeds

def upcast_vae(model):
    from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor

    dtype = model.dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        new_dtype = torch.bfloat16
    else:
        new_dtype = torch.float32

    model = model.to(dtype=new_dtype)
    use_torch_2_0_or_xformers = isinstance(
        model.decoder.mid_block.attentions[0].processor,
        (
            AttnProcessor2_0,
            XFormersAttnProcessor,
        ),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        model.post_quant_conv.to(dtype)
        model.decoder.conv_in.to(dtype)
        model.decoder.mid_block.to(dtype)

    return model

def sd3_latents_to_rgb(latents: torch.Tensor):
    if latents is None:
        return None
    
    if latents.dim() == 4:
        latents = latents[0]

    scale_factor = 1.5305
    shift_factor = 0.0609

    # The SD3 latent_rgb_factors matrix
    latent_rgb_factors = torch.tensor([
        [-0.0645,  0.0177,  0.1052],
        [ 0.0028,  0.0312,  0.0650],
        [ 0.1848,  0.0762,  0.0360],
        [ 0.0944,  0.0360,  0.0889],
        [ 0.0897,  0.0506, -0.0364],
        [-0.0020,  0.1203,  0.0284],
        [ 0.0855,  0.0118,  0.0283],
        [-0.0539,  0.0658,  0.1047],
        [-0.0057,  0.0116,  0.0700],
        [-0.0412,  0.0281, -0.0039],
        [ 0.1106,  0.1171,  0.1220],
        [-0.0248,  0.0682, -0.0481],
        [ 0.0815,  0.0846,  0.1207],
        [-0.0120, -0.0055, -0.0867],
        [-0.0749, -0.0634, -0.0456],
        [-0.1418, -0.1457, -0.1259]
    ], dtype=latents.dtype, device=latents.device)

    latents = latents.permute(1, 2, 0)
    latents = (latents - shift_factor) / scale_factor

    # Perform the linear transformation
    rgb_pixels = latents @ latent_rgb_factors

    rgb_pixels = rgb_pixels.permute(2, 0, 1)

    rgb_pixels = rgb_pixels.float()

    # Clamp values to the global percentile range and normalize
    q_005 = torch.quantile(rgb_pixels, 0.005)
    q_995 = torch.quantile(rgb_pixels, 0.995)
    image_tensor = torch.clamp(rgb_pixels, q_005, q_995)
    image_tensor = (image_tensor - q_005) / (q_995 - q_005).add(1e-6)

    image = image_tensor.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    
    image = Image.fromarray(image)
    return image