import torch

def device_list():
    devices = {}

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            devices[f'cuda:{i}'] = {
                'name': f'{torch.cuda.get_device_name(i)} {total_memory / 1024 ** 3:.2f}GB ({i})',
                'label': [f'cuda:{i}'],
                'total_memory': total_memory,
                'index': i,
            }

    if torch.mps.is_available():
        for i in range(torch.mps.device_count()):
            devices[f'mps:{i}'] = {
                'name': f'MPS ({i})',
                'label': [f'mps:{i}'],
                'total_memory': 0,
                'index': i,
            }
    
    if torch.cpu.is_available():
        for i in range(torch.cpu.device_count()):
            devices[f'cpu:{i}'] = {
                'name': f'CPU ({i})',
                'label': [f'cpu:{i}'],
                'total_memory': 0,
                'index': i,
            }

    return devices

DEVICE_LIST = device_list()
DEFAULT_DEVICE = list(DEVICE_LIST.keys())[0]

def str_to_dtype(dtype, *args):
    return {
        'auto': None,
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8_e4m3fn': torch.float8_e4m3fn,
    }[dtype]

def compile(model):
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True
    model.to(memory_format=torch.channels_last)

    return torch.compile(model, mode='max-autotune', fullgraph=True)

def TensorToImage(tensor):
    from torchvision.transforms import v2 as tt
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    return [tt.ToPILImage()(t.clamp(0, 1).float()) for t in tensor]