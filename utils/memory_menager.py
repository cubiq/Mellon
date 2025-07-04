import logging
logger = logging.getLogger('mellon')
import torch
import gc
import time
import nanoid
from utils.torch_utils import DEFAULT_DEVICE

def memory_flush():
    gc.collect()

    if torch.cuda.is_available():
        #torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.mps.is_available():
        #torch.mps.synchronize()
        torch.mps.empty_cache()

class MemoryManager:
    def __init__(self):
        self.cache = {}
    
    def add(self, model, priority=1) -> str:
        device = model.device if hasattr(model, 'device') else 'cpu'

        if hasattr(model, '_mm_id'):
            model_id = model._mm_id
            if model_id in self.cache:
                self.update(model_id, model=model, priority=priority, device=device)
                return model_id
            else:
                model.to('cpu')

        model_id = nanoid.generate()
        model._mm_id = model_id
        self.cache[model_id] = {
            'model': model,
            'device': device,
            'priority': priority,
            'last_used': time.time(),
            'size': 0
        }
        return model_id
    
    def remove(self, model):
        model_id = model if isinstance(model, str) else model._mm_id if hasattr(model, '_mm_id') else None
        if model_id is None or model_id not in self.cache:
            return None
        
        self.cache[model_id]['model'] = self.cache[model_id]['model'].to('cpu')
        self.cache[model_id]['model'] = None
        del self.cache[model_id]
        memory_flush()
        return model_id
    
    def update(self, model, **kwargs):
        model_id = model if isinstance(model, str) else model._mm_id if hasattr(model, '_mm_id') else None
        if model_id is None or model_id not in self.cache:
            return None
        
        for k, v in kwargs.items():
            self.cache[model_id][k] = v
        
        self.cache[model_id]['last_used'] = time.time()
        memory_flush()
        return model_id
    
    def get_model(self, model_id):
        if model_id not in self.cache:
            return None

        return self.cache[model_id]['model']

    def load_model(self, model, device, exclude=[]):
        model_id = model if isinstance(model, str) else model._mm_id if hasattr(model, '_mm_id') else None
        if model_id is None or model_id not in self.cache:
            return None

        exclude_ids = []
        for v in exclude:
            k = v if isinstance(v, str) else v._mm_id if hasattr(v, '_mm_id') else None
            if k and k in self.cache:
                exclude_ids.append(k)
        exclude_ids.append(model_id)
    
        self.cache[model_id]['last_used'] = time.time()
        model = self.cache[model_id]['model']

        if model.device == device:
            return model
        
        cache_priority = self._get_unload_candidates(device, exclude_ids)

        memory_flush()

        # First we try to unload models based on the size if we have set it
        # memory_required = self.cache[model_id]['size']
        # memory_available = torch.cuda.mem_get_info()[0] if 'cuda' in device else 0
        # if memory_available > 0 and memory_required > memory_available and all(v['size'] > 0 for v in self.cache.values()):
        #     while memory_required > memory_available and cache_priority:
        #         k = cache_priority.pop(0)[3]
        #         logger.debug(f"Unloading model {k} to free memory. Memory available: {memory_available}, memory required: {memory_required}")
        #         self.unload_model(k)
        #         memory_available = torch.cuda.mem_get_info()[0]

        while True:
            try:
                memory_current = torch.cuda.mem_get_info()[0] if 'cuda' in device else 0
                print(f"Memory current: {memory_current / 1024 / 1024} MB")
                model = model.to(device)
                self.cache[model_id]['model'] = model
                self.cache[model_id]['device'] = device
                if self.cache[model_id]['size'] == 0 and 'cuda' in device:
                    self.cache[model_id]['size'] = memory_current - torch.cuda.mem_get_info()[0]
                return model
            except torch.OutOfMemoryError as e:
                if not cache_priority:
                    logger.debug(f"Cannot free enough memory to load model {model_id}")
                    raise e
                
                k = cache_priority.pop(0)[3]
                logger.debug(f"OOM error, unloading lower priority model: {k}")
                self.unload_model(k)
            except Exception as e:
                logger.error(f"Error loading model {model_id}")
                raise e
    
    def unload_model(self, model):
        model_id = model if isinstance(model, str) else model._mm_id if hasattr(model, '_mm_id') else None
        if model_id is None or model_id not in self.cache:
            return None
        print(f"Unloading model {model_id}")
        model = self.cache[model_id]['model'].to('cpu')
        self.cache[model_id]['model'] = None
        self.cache[model_id]['model'] = model
        self.cache[model_id]['device'] = 'cpu'
        memory_flush()

        return self.cache[model_id]['model']
    
    def unload_all(self, device=None):
        device = device if device else DEFAULT_DEVICE
        for k, v in self.cache.items():
            if str(v['device']) == str(device):
                self.unload_model(k)
    
    def exec(self, func, device, exclude=[], args=None, kwargs=None):
        exclude_ids = []
        for v in exclude:
            k = v if isinstance(v, str) else v._mm_id if hasattr(v, '_mm_id') else None
            if k and k in self.cache:
                exclude_ids.append(k)
        
        # Get a list of all models on the target device that can be unloaded.
        cache_priority = self._get_unload_candidates(device, exclude_ids=exclude_ids)

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        while True:
            try:
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                # If we're out of memory, we need to unload a model.
                if not cache_priority:
                    # If there are no more models to unload, we have failed.
                    logger.error(f"OOM during exec. No models left to unload to free memory.")
                    raise e
                
                # Unload the lowest-priority model.
                k = cache_priority.pop(0)[3]
                logger.debug(f"OOM during exec. Unloading model '{k}' to free VRAM.")
                self.unload_model(k)
            except Exception as e:
                logger.error(f"An unexpected error occurred during exec: {e}")
                raise e
        
    def _get_unload_candidates(self, device, exclude_ids=[]):
        """
        Gets a sorted list of models that are candidates for unloading from a device.
        The list is sorted by priority and last-used time, so the first element
        is the best candidate for unloading.
        """
        cache_priority = []
        for k, v in self.cache.items():
            # Check if the model is on the target device and not in the exclude list.
            if str(v['device']) == str(device) and k not in exclude_ids:
                cache_priority.append((v['priority'], v['last_used'], v['size'], k))
        
        # Sort by priority (asc) then last_used (asc) to find the best unload candidate first.
        cache_priority.sort(key=lambda x: (x[0], x[1]))
        return cache_priority

memory_manager = MemoryManager()