def getQuantizationConfig(method, **kwargs):
    weights = kwargs.get('weights', None)
    dtype = kwargs.get('dtype', None)

    if method == 'bitsandbytes':
        weights = kwargs.get('bnb_type', '4bit')
        double_quant = kwargs.get('bnb_double_quant', True)
        return getBnBConfig(weights, dtype=dtype, double_quant=double_quant)
    elif method == 'torchao':
        quant_type = kwargs.get('torchao_quant_type', 'float8wo_e4m3')
        return getTorchAOConfig(quant_type)
    elif method == 'quanto':
        weights = kwargs.get('quanto_type', 'float8')
        return getQuantoConfig(weights)

    return None

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

def getTorchAOConfig(quant_type):
    from diffusers import TorchAoConfig

    config = TorchAoConfig(quant_type=quant_type)

    return config

def getQuantoConfig(weights):
    from diffusers import QuantoConfig

    config = QuantoConfig(weights=weights)

    return config
