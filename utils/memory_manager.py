import torch
import gc
import time
from utils.torch_utils import device_list
from enum import Enum
import logging
logger = logging.getLogger('mellon')


def memory_flush(gc_collect=False, reset=False):
    if gc_collect:
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if reset:
            for _, d in device_list.items():
                torch.cuda.reset_max_memory_allocated(d['index'])
                torch.cuda.reset_peak_memory_stats(d['index'])

class MemoryManager:
    def __init__(self, memory_threshold=.9):
        self.cache = {}
        self.memory_threshold = memory_threshold

    def add_model(self, model, model_id, device='cpu', priority=2):
        priority = priority if isinstance(priority, int) else 2

        if model_id not in self.cache:
            self.cache[model_id] = {
                'model': model,
                'device': device,           # device the model is currently on
                'priority': priority,       # priority, lower priority models are unloaded first
                'last_used': time.time(),   # time the model was last used
            }

        return model_id

    def get_available_memory(self, device):
        return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)

    def get_model(self, model_id):
        return self.cache[model_id]['model'] if model_id in self.cache else None

    def get_model_info(self, model_id):
        return self.cache[model_id] if model_id in self.cache else None

    def load_model(self, model_id, device):
        self.cache[model_id]['last_used'] = time.time()
        x = self.cache[model_id]['model']

        if device == str(x.device):
            return x

        if device == 'cpu':
            return self.unload_model(model_id)

        cache_priority = []
        # Sort models by priority and last_used
        for id, model in self.cache.items():
            if model['device'] == device:
                cache_priority.append((model['priority'], model['last_used'], id))

        cache_priority.sort()
        memory_flush()

        while True:
            # Attempt to load the model
            try:
                x = x.to(device)
                self.cache[model_id]['device'] = device
                return x
            
            except torch.OutOfMemoryError as e:
                if not cache_priority:
                    logger.debug("No more models to unload, cannot free sufficient memory")
                    raise e

                next_model_id = cache_priority.pop(0)[2]
                logger.debug(f"OOM error, unloading lower priority model: {next_model_id}")
                self.unload_model(next_model_id)

            except Exception as e:
                raise e


    def unload_model(self, model_id):
        if model_id in self.cache and hasattr(self.cache[model_id]['model'], 'to'):
            model = self.cache[model_id]['model'].to('cpu')
            self.cache[model_id]['model'] = None
            self.cache[model_id]['model'] = model
            self.cache[model_id]['device'] = 'cpu'
            memory_flush()

        return self.cache[model_id]['model']
    
    def unload_all(self, exclude=[]):
        if not isinstance(exclude, list):
            exclude = [exclude]

        for model_id in self.cache:
            if model_id not in exclude:
                self.unload_model(model_id)

    def delete_model(self, model_id, unload=False):
        model_id = model_id if isinstance(model_id, list) else [model_id]

        for m in model_id:
            if m in self.cache:
                classname = self.cache[m]['model'].__class__.__name__ if hasattr(self.cache[m]['model'], '__class__') else 'Unknown'
                logger.debug(f"Deleting model {classname}, id: {m}")
                if unload:
                    self.unload_model(m)
                self.cache[m]['model'] = None
                del self.cache[m]

        memory_flush(gc_collect=True)

    def update_model(self, model_id, model=None, priority=None, unload=True):
        if model_id in self.cache:
            if model:
                if unload:
                    self.unload_model(model_id)
                self.cache[model_id]['model'] = model
                memory_flush()
            if priority:
                self.cache[model_id]['priority'] = priority

    def is_cached(self, model_id):
        return model_id in self.cache
    
    def cache_count(self):
        return len(self.cache)
    
    def flash_load(self, model, model_id, device='cpu', priority=3):
        model_id = self.add_model(model, model_id, device=device, priority=priority)
        model = self.load_model(model_id, device)
        self.delete_model(model_id)

        return model
    
    def unload_next(self, device, exclude=[]):
        if not self.cache:
            return False
        
        if not isinstance(exclude, list):
            exclude = [exclude]

        # Sort models by priority and last_used
        cache_priority = []
        for id, model in self.cache.items():
            if model['device'] == device and id not in exclude:
                cache_priority.append((model['priority'], model['last_used'], id))

        cache_priority.sort()
        next_model_id = cache_priority.pop(0)[2]

        self.unload_model(next_model_id)
        return True

memory_manager = MemoryManager()
