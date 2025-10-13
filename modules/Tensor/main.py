import torch
from mellon.NodeBase import NodeBase

class AddTensorNoise(NodeBase):
    label = "Add Tensor Noise"
    category = "primitive"
    params = {
        "tensor": { "label": "Tensor", "type": "any", "display": "input" },
        "noisy_tensor": { "label": "Noisy Tensor", "type": "any", "display": "output" },
        "noise_level": { "label": "Noise Level", "type": "float", "display": "slider", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01 },
        "seed": { "label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295 },
    }

    def execute(self, tensor, **kwargs):
        if tensor is None:
            return { "noisy_tensor": None }
        
        noise_level = kwargs.get("noise_level", 0.1)
        seed = kwargs.get("seed", 0)

        if noise_level <= 0:
            return { "noisy_tensor": tensor }

        def add_noise(t):
            if not isinstance(t, torch.Tensor):
                return t
            generator = torch.Generator(device=t.device).manual_seed(seed)
            #noise = torch.randn_like(t, generator=generator) * noise_level
            noise = torch.empty_like(t).normal_(generator=generator) * noise_level
            return t + noise

        def process_input(inp):
            if isinstance(inp, torch.Tensor):
                return add_noise(inp)
            elif isinstance(inp, dict):
                return {k: process_input(v) for k, v in inp.items()}
            elif isinstance(inp, list):
                return [process_input(item) for item in inp]
            else:
                return inp

        noisy_tensor = process_input(tensor)
        return { "noisy_tensor": noisy_tensor }

class MergeTensors(NodeBase):
    label = "Merge Tensors"
    category = "primitive"
    params = {
        "tensor_1": { "label": "Tensor 1", "type": "any", "display": "input" },
        "tensor_2": { "label": "Tensor 2", "type": "any", "display": "input" },
        "merged_tensor": { "label": "Merged Tensor", "type": "any", "display": "output" },
        "factor": { "label": "Factor", "type": "float", "display": "slider", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01 },
    }

    def execute(self, tensor_1, tensor_2, **kwargs):
        if tensor_1 is None or tensor_2 is None:
            return { "merged_tensor": None }
        
        factor = kwargs.get("factor", 0.5)
        factor = max(0.0, min(1.0, factor))

        if factor == 0.0:
            return { "merged_tensor": tensor_1 }
        if factor == 1.0:
            return { "merged_tensor": tensor_2 }

        def merge(t1, t2):
            if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
                return t1 if t1 is not None else t2
            if t1.shape != t2.shape:
                return t1
            #return t1 * (1 - factor) + t2 * factor
            return torch.lerp(t1, t2, factor)

        def process_inputs(inp1, inp2):
            if isinstance(inp1, torch.Tensor) and isinstance(inp2, torch.Tensor):
                return merge(inp1, inp2)
            elif isinstance(inp1, dict) and isinstance(inp2, dict):
                keys = set(inp1.keys()).union(set(inp2.keys()))
                return {k: process_inputs(inp1.get(k), inp2.get(k)) for k in keys}
            elif isinstance(inp1, list) and isinstance(inp2, list):
                length = min(len(inp1), len(inp2))
                return [process_inputs(inp1[i], inp2[i]) for i in range(length)]
            else:
                return inp1 if inp1 is not None else inp2

        merged_tensor = process_inputs(tensor_1, tensor_2)
        return { "merged_tensor": merged_tensor }

class AverageTensors(NodeBase):
    label = "Average Tensors"
    category = "primitive"
    params = {
        "tensors": { "label": "Tensor", "type": "any", "display": "input", "spawn": True },
        "averaged_tensor": { "label": "Averaged Tensor", "type": "any", "display": "output" },
    }

    def execute(self, tensors, **kwargs):
        if not tensors:
            return { "averaged_tensor": None }
        
        if not isinstance(tensors, list):
            tensors = [tensors]
        
        first = tensors[0]
        
        if isinstance(first, torch.Tensor):
            # List of tensors
            tensor_list = [t for t in tensors if isinstance(t, torch.Tensor)]
            if not tensor_list:
                return { "averaged_tensor": None }
            return { "averaged_tensor": torch.mean(torch.stack(tensor_list), dim=0) }
        
        elif isinstance(first, dict):
            # List of dicts
            result = {}
            keys = set()
            for d in tensors:
                if isinstance(d, dict):
                    keys.update(d.keys())
            
            for key in keys:
                tensor_list = [d[key] for d in tensors if isinstance(d, dict) and key in d and isinstance(d[key], torch.Tensor)]
                if tensor_list:
                    result[key] = torch.mean(torch.stack(tensor_list), dim=0)
            return { "averaged_tensor": result }
        
        elif isinstance(first, list):
            # List of lists
            max_len = max((len(l) for l in tensors if isinstance(l, list)), default=0)
            result = []
            for i in range(max_len):
                tensor_list = []
                for l in tensors:
                    if isinstance(l, list) and i < len(l) and isinstance(l[i], torch.Tensor):
                        tensor_list.append(l[i])
                if tensor_list:
                    result.append(torch.mean(torch.stack(tensor_list), dim=0))
                else:
                    result.append(None)
            return { "averaged_tensor": result }
        
        return { "averaged_tensor": None }
    
class MultiplyTensor(NodeBase):
    label = "Multiply Tensor"
    category = "primitive"
    params = {
        "tensor": { "label": "Tensor", "type": "any", "display": "input" },
        "output": { "label": "Output", "type": "any", "display": "output" },
        "factor": { "label": "Factor", "type": "float", "display": "slider", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 },
    }

    def execute(self, tensor, **kwargs):
        if tensor is None:
            return { "output": None }
        
        factor = kwargs.get("factor", 1.0)

        if factor == 1.0:
            return { "output": tensor }

        def multiply(t):
            if not isinstance(t, torch.Tensor):
                return t
            return t * factor

        def process_input(inp):
            if isinstance(inp, torch.Tensor):
                return multiply(inp)
            elif isinstance(inp, dict):
                return {k: process_input(v) for k, v in inp.items()}
            elif isinstance(inp, list):
                return [process_input(item) for item in inp]
            else:
                return inp

        multiplied_tensor = process_input(tensor)
        return { "output": multiplied_tensor }