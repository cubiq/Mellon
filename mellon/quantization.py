from config import config
from utils.memory_manager import memory_manager
import os

def set_compile_env():
    if 'CC' in config.environ and config.environ['CC']:
        os.environ['CC'] = config.environ['CC']
    if 'CXX' in config.environ and config.environ['CXX']:
        os.environ['CXX'] = config.environ['CXX']
    if 'TORCH_CUDA_ARCH_LIST' in config.environ and config.environ['TORCH_CUDA_ARCH_LIST']:
        os.environ['TORCH_CUDA_ARCH_LIST'] = config.environ['TORCH_CUDA_ARCH_LIST']

def quanto(model, weights, activations=None, exclude=None, device=None):
    from optimum.quanto import freeze, quantize

    set_compile_env()

    if device:
        model.to(device)

    weights_dtype = f"q{weights.lower()}"
    activations_dtype = f"q{activations.lower()}" if activations != 'none' else None

    weights_module = getattr(__import__('optimum.quanto', fromlist=[weights_dtype]), weights_dtype)
    activations_module = None
    if activations_dtype:
        activations_module = getattr(__import__('optimum.quanto', fromlist=[activations_dtype]), activations_dtype)

    exclude = exclude or []
    if isinstance(exclude, str):
        exclude = [item.strip() for item in exclude.split(',')]

    quantize(model, weights=weights_module, activations=activations_module, exclude=exclude)
    freeze(model)

    return model

def torchao(model, weights, device=None):
    from torchao.quantization import quantize_

    set_compile_env()

    if weights == 'fp6':
        from torchao.quantization import fpx_weight_only
        quantize_(model, fpx_weight_only(3, 2), device=device)
    else:
        weights_dtype = f"{weights.lower()}"
        weights_module = getattr(__import__('torchao.quantization.quant_api', fromlist=[weights_dtype]), weights_dtype)
        quantize_(model, weights_module(), device=device)

    return model

def bitsandbytes(weights, dtype=None, double_quant=False):
    from diffusers import BitsAndBytesConfig

    if weights == '8-bit':
        config = BitsAndBytesConfig(load_in_8bit=True)
    elif weights == '4-bit':
        config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=double_quant,
                                    bnb_4bit_compute_dtype=dtype)

    return config

class NodeQuantization():
    def quantize(self, type, model=None, **kwargs):
        model_id = model if isinstance(model, str) else model.get('_mm_id', None)
        if not model_id:
            raise ValueError("Model ID is required for quantization")

        if type == 'none':
            return model
        elif type == 'torchao':
            return self._torchao(model=model_id, **kwargs)
        elif type == 'quanto':
            return self._quanto(model=model_id, **kwargs)
        else:
            raise ValueError(f"Invalid quantization type: {type}")

    def _torchao(self, model=None, torchao_device=None, torchao_weights=None, torchao_individual_layers=False, **kwargs):       
        device = torchao_device if torchao_individual_layers else None

        if not torchao_individual_layers:
            memory_manager.unload_all(exclude=[model])
            self.mm_load(model, device)
        else:
            memory_manager.unload_all()

        model = torchao(self.mm_get(model), torchao_weights, device=device)
        self.mm_update(model, model=model)
        return model

    def _quanto(self, model=None, quanto_device=None, quanto_weights=None, quanto_activations=None, quanto_exclude=None, **kwargs):
        memory_manager.unload_all(exclude=[model])
        self.mm_load(model, quanto_device)
        model = quanto(self.mm_get(model), quanto_weights, activations=quanto_activations, exclude=quanto_exclude)
        self.mm_update(model, model=model)
        return model
    

