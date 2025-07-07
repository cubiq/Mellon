import torch

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