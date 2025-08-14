def getQuantizationConfig(method, **kwargs):
    #weights = kwargs.get('weights', None)
    bnb_type = kwargs.get('bnb_type', '4bit')
    dtype = kwargs.get('dtype', None)

    if method == 'bnb':
        double_quant = kwargs.get('bnb_double_quant', True)
        return getBnBConfig(bnb_type, dtype=dtype, double_quant=double_quant)
    elif method == 'torchao':
        quant_type = kwargs.get('torchao_quant_type', 'float8wo_e4m3')
        return getTorchAOConfig(quant_type)
    elif method == 'quanto':
        weights = kwargs.get('quanto_type', 'float8')
        return getQuantoConfig(weights)

    return None

def getBnBConfig(bnb_type, dtype=None, double_quant=True):
    from diffusers import BitsAndBytesConfig

    if bnb_type == '8bit':
        config = BitsAndBytesConfig(load_in_8bit=True)
    elif bnb_type == '4bit':
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

def quantize(model, method, **kwargs):
    quanto_weights = kwargs.get('quanto_weights', 'qfloat8')
    torchao_quant_type = kwargs.get('torchao_quant_type', 'float8wo_e4m3')
    activations = kwargs.get('activations', None)
    exclude = kwargs.get('quant_exclude', None)

    if method == 'torchao':
        torchao(model, quant_type=torchao_quant_type)
    elif method == 'quanto':
        quanto(model, weights=quanto_weights, activations=activations, exclude=exclude)
    
    return model

def torchao(model, quant_type):
    from torchao.quantization import quantize_

    supported_quant_types = [
        'int4wo', 'int4dq', 'int8wo', 'int8dq', 'int8dq_int4w',
        'uint1wo', 'uint2wo', 'uint3wo', 'uint4wo', 'uint5wo', 'uint6wo', 'uint7wo',
        'float8wo_e5m2', 'float8wo_e4m3', 'float8dq_e4m3', 'float8dq_e4m3_tensor', 'float8dq_e4m3_row',
        'fp3_e1m1', 'fp3_e2m0', 'fp4_e1m2', 'fp4_e2m1', 'fp4_e3m0', 'fp5_e1m3', 'fp5_e2m2',
        'fp5_e3m1', 'fp5_e4m0', 'fp6_e1m4', 'fp6_e2m3', 'fp6_e3m2', 'fp6_e4m1', 'fp6_e5m0',
        'fp7_e1m5', 'fp7_e2m4', 'fp7_e3m3', 'fp7_e4m2', 'fp7_e5m1', 'fp7_e6m0'
    ]
    quant_type = quant_type if quant_type in supported_quant_types else 'float8wo_e4m3'
    dtype = get_torchao_quant_method(quant_type)
    quantize_(model, dtype())
    return model

def quanto(model, weights, activations=None, exclude=None):
    from optimum.quanto import freeze, quantize, qfloat8, qint8, qint4, qint2

    weights_map = { 'float8': qfloat8, 'int8': qint8, 'int4': qint4, 'int2': qint2 }
    weights = weights.lower()
    weights = weights_map.get(weights, qfloat8)

    activations_map = { 'float8': qfloat8, 'int8': qint8 }
    activations = activations_map.get(activations) if activations else None

    if exclude is None:
        exclude = []

    quantize(model, weights=weights, activations=activations, exclude=exclude)
    freeze(model)

    return model


def get_torchao_quant_method(quant_type):
    import torch
    from functools import partial
    from torchao.quantization import (
        float8_dynamic_activation_float8_weight,
        float8_static_activation_float8_weight,
        float8_weight_only,
        fpx_weight_only,
        int4_weight_only,
        int4_dynamic_activation_int4_weight,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        uintx_weight_only,
    )
    from torchao.quantization.observer import PerRow, PerTensor

    # Integer quantization
    int_map = {
        'int4wo': int4_weight_only,
        'int4dq': int4_dynamic_activation_int4_weight,
        'int8wo': int8_weight_only,
        'int8dq': int8_dynamic_activation_int8_weight,
    }
    if quant_type in int_map:
        return int_map[quant_type]

    # Floating point 8-bit quantization
    float8_map = {
        'float8wo': float8_weight_only,
        'float8sq': float8_static_activation_float8_weight,
        'float8wo_e5m2': partial(float8_weight_only, weight_dtype=torch.float8_e5m2),
        'float8wo_e4m3': partial(float8_weight_only, weight_dtype=torch.float8_e4m3fn),
        'float8dq': float8_dynamic_activation_float8_weight,
        'float8dq_e4m3': partial(float8_dynamic_activation_float8_weight, activation_dtype=torch.float8_e4m3fn, weight_dtype=torch.float8_e4m3fn),
        'float8dq_e4m3_tensor': partial(float8_dynamic_activation_float8_weight, activation_dtype=torch.float8_e4m3fn, weight_dtype=torch.float8_e4m3fn, granularity=PerTensor()),
        'float8dq_e4m3_row': partial(float8_dynamic_activation_float8_weight, activation_dtype=torch.float8_e4m3fn, weight_dtype=torch.float8_e4m3fn, granularity=PerRow())
    }
    if quant_type in float8_map:
        return float8_map[quant_type]

    # Floating point X-bit quantization
    import re
    fp_match = re.match(r'fp(\d)_e(\d+)m(\d+)', quant_type)
    if fp_match:
        X, A, B = int(fp_match.group(1)), int(fp_match.group(2)), int(fp_match.group(3))
        if X == A + B + 1:
            return partial(fpx_weight_only, A, B)

    # Unsigned integer quantization
    uint_match = re.match(r'uint(\d)wo', quant_type)
    if uint_match:
        X = int(uint_match.group(1))
        dtype = getattr(torch, f'uint{X}', 'uint7')
        return partial(uintx_weight_only, dtype=dtype)

    return None