import torch

def list_devices():
    host = ""  # TODO: support multiple hosts

    devices = {}
    default_device = None
    cpu = {
        "index": 0,
        "device": "cpu",
        "host": host,
        "label": host + "cpu",
        "total_memory": None,
        #"name": "CPU"
    } # TODO: probably need to support multiple cpus

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            key = f"{host}cuda:{i}"

            if default_device is None:
                default_device = key

            devices[key] = {
                "index": i,
                "device": f"cuda:{i}",
                "host": host,
                "label": host + f"cuda:{i}",
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
                #"name": f"{torch.cuda.get_device_properties(i).name}"
            }

        devices[f"{host}cpu"] = cpu

    elif torch.mps.is_available():
        key = f"{host}mps"
        default_device = key
        devices[key] = {
            "index": 0,
            "device": "mps",
            "host": host,
            "label": host + "mps",
            "total_memory": None,
            #"name": "MPS"
        }

    else:
        key = f"{host}cpu"
        default_device = key
        devices[key] = cpu

    return devices, default_device

device_list, default_device = list_devices()

def str_to_dtype(dtype, params):
    return {
        'auto': None,
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8_e4m3fn': torch.float8_e4m3fn,
    }[dtype]

def toTensor(image):
    from torchvision.transforms import v2 as tt
    image = tt.PILToTensor()(image) / 255.0
    return image

def toPIL(tensor):
    from torchvision.transforms import v2 as tt
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    images = []
    for t in tensor:
        images.append(tt.ToPILImage()(t.clamp(0, 1).float()))

    return images

def toLatent(image):
    from torchvision.transforms import v2 as tt
    image = image.convert('RGB')
    image = tt.PILToTensor()(image) / 127.5 - 1
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    return image

def compile(model):
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True
    model.to(memory_format=torch.channels_last)

    return torch.compile(model, mode='max-autotune', fullgraph=True)
