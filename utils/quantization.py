def getBnBConfig(weights, dtype=None, double_quant=False):
    from diffusers import BitsAndBytesConfig

    if weights == '8bit':
        config = BitsAndBytesConfig(load_in_8bit=True)
    elif weights == '4bit':
        config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=double_quant,
                                    bnb_4bit_compute_dtype=dtype)

    return config