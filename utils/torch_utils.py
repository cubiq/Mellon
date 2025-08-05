import torch

def device_list():
    devices = {}

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            devices[f'cuda:{i}'] = {
                'arch': 'cuda',
                'name': f'{torch.cuda.get_device_name(i)} {total_memory / 1024 ** 3:.2f}GB ({i})',
                'label': [f'cuda:{i}'],
                'total_memory': total_memory,
                'index': i,
            }

    if torch.mps.is_available():
        for i in range(torch.mps.device_count()):
            devices[f'mps:{i}'] = {
                'arch': 'mps',
                'name': f'MPS ({i})',
                'label': [f'mps:{i}'],
                'total_memory': 0,
                'index': i,
            }
    
    if torch.cpu.is_available():
        for i in range(torch.cpu.device_count()):
            devices[f'cpu:{i}'] = {
                'arch': 'cpu',
                'name': f'CPU ({i})',
                'label': [f'cpu:{i}'],
                'total_memory': 0,
                'index': i,
            }

    return devices

DEVICE_LIST = device_list()
DEFAULT_DEVICE = list(DEVICE_LIST.keys())[0]
CPU_DEVICE = 'cpu:0' if 'cpu:0' in DEVICE_LIST else DEFAULT_DEVICE
IS_CUDA = any('cuda' in device for device in DEVICE_LIST.keys())

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

    tensor = tensor if isinstance(tensor, list) else [tensor]
    output = []
    for t in tensor:
        if t.ndim == 4:
            t = t.squeeze(0)
        output.append(tt.ToPILImage()(t.clamp(0, 1).float()))

    return output

def ImageToTensor(image):
    from torchvision.transforms import v2 as tt
    #return tt.ToTensor()(image)
    return tt.Compose([
        tt.ToImage(),
        tt.ToDtype(torch.float32, scale=True)
    ])(image)

def get_memory_stats():
    if not torch.cuda.is_available():
        return {}
    
    stats = torch.cuda.memory_stats()

    return {
        'current': stats['allocated_bytes.all.current'],
        'peak': stats['allocated_bytes.all.peak'],
        'allocated': stats['allocated_bytes.all.allocated'],
        'freed': stats['allocated_bytes.all.freed'],
    }

def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()