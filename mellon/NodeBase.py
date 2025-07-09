import logging
logger = logging.getLogger('mellon')
from modules import MODULE_MAP
from mellon.server import server
from utils.memory_menager import memory_manager, memory_flush
import numpy as np
import torch
import sys

def get_module_output(module_name, class_name):
    params = MODULE_MAP[module_name][class_name]['params'] if module_name in MODULE_MAP and class_name in MODULE_MAP[module_name] else {}
    return { p: None for p in params if 'display' in params[p] and params[p]['display'] == 'output' }

def get_default_params(module_name, class_name):
    params = MODULE_MAP[module_name][class_name]['params'] if module_name in MODULE_MAP and class_name in MODULE_MAP[module_name] else {}
    return { k: v for k, v in params.items() if not 'display' in v or v['display'] not in ['output', 'ui'] }

def deep_equal(a, b):
    # 1. Identity check
    if a is b:
        return True

    # 2. Type check
    if type(a) is not type(b):
        return False
    
    # 3. Specific type checks
    if isinstance(a, torch.Tensor):
        if a.device != b.device or a.dtype != b.dtype or a.shape != b.shape:
            return False
        return torch.equal(a, b)
    
    # For numpy arrays, check dtype, shape, and then values.
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        return np.array_equal(a, b)

    # For PIL-like images (duck-typing)
    if hasattr(a, 'getdata') and callable(a.getdata) and hasattr(b, 'getdata') and callable(b.getdata):
        if a.size != b.size or a.mode != b.mode:
            return False
        #return list(a.getdata()) == list(b.getdata())
        return np.array_equal(np.array(a), np.array(b)) # this should be faster than the above

    # For lists and tuples, check length and then each element recursively.
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))
        
    # For sets, the standard equality check is sufficient.
    if isinstance(a, set):
        return a == b

    # For dictionaries, check keys and then each value recursively.
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[key], b[key]) for key in a)

    # 4. Generic object checks
    # If objects have a 'to_dict' method, compare their dict representations.
    if hasattr(a, 'to_dict') and callable(a.to_dict):
        return deep_equal(a.to_dict(), b.to_dict())

    # As a fallback, compare the objects' __dict__ attributes recursively.
    if hasattr(a, '__dict__'):
        return deep_equal(a.__dict__, b.__dict__)

    # 5. Default case: use standard equality for primitives (int, str, etc.)
    return a == b

class NodeBase:
    CALLBACK = 'execute'

    def __init__(self, node_id=None):
        self.node_id = node_id
        # take everything before the last dot
        self.module_name = ".".join(self.__class__.__module__.split(".")[:-1])
        self.class_name = self.__class__.__name__

        self.params = {}
        self.default_params = get_default_params(self.module_name, self.class_name)
        self.output = get_module_output(self.module_name, self.class_name)

        self._sid = None
        self._execution_time = { 'last': None, 'min': None, 'max': None }
        self._mm_models = []
        self._interrupt = False

    def __call__(self, **kwargs):
        self._interrupt = False
        
        # filter out params that are not in the default_params
        params = { key: kwargs[key] for key in kwargs if key in self.default_params }

        # if node_id is None, the class was called directly, so we execute it without further processing
        if self.node_id is None:
            return getattr(self, self.CALLBACK)(**params)
        
        # params normalization and validation
        for key, value in params.items():
            if value is None:
                value = self.default_params[key]['default'] if 'default' in self.default_params[key] else None
                params[key] = value
            
            if 'type' in self.default_params[key]:
                type = self.default_params[key]['type']
                if isinstance(type, list):
                    type = type[0]

                if type.startswith('int'):
                    params[key] = int(value or 0) if not isinstance(value, list) else [int(v) for v in value]
                elif type.startswith('float'):
                    params[key] = float(value or 0) if not isinstance(value, list) else [float(v) for v in value]
                elif type.startswith('str') or type.startswith('text'):
                    if isinstance(value, dict):
                        params[key] = value
                    elif isinstance(value, list):
                        params[key] = [str(v) for v in value]
                    else:
                        params[key] = str(value or '')
                elif type.startswith('bool'):
                    params[key] = bool(value) if not isinstance(value, list) else [bool(v) for v in value]
            
            if 'options' in self.default_params[key] and not ('no_validation' in self.default_params[key] and self.default_params[key]['no_validation']):
                options = self.default_params[key]['options']
                value_list = [value] if not isinstance(value, list) else value
                if isinstance(options, list):
                    if any(v not in options for v in value_list):
                        raise ValueError(f"Module {self.module_name}.{self.class_name}: Invalid value for {key}: {value} (options: {options})")
                elif isinstance(options, dict):
                    if any(v not in options.keys() for v in value_list):
                        raise ValueError(f"Module {self.module_name}.{self.class_name}: Invalid value for {key}: {value} (options: {options})")
                else:
                    raise ValueError(f"Module {self.module_name}.{self.class_name}: Invalid options format for {key}: {options}")
        
        # post processing
        for key in self.default_params:
            if 'postProcess' in self.default_params[key]:
                # we pass the current value and the dict of all the values for cross parameter validation
                params[key] = self.default_params[key]['postProcess'](params[key], params)
        
        # if any of the values has changed or self.output is empty, we need to execute the node
        if (not deep_equal(self.params, params)) or any(v is None for v in self.output.values()):
            self.params = params
            self.output = {k: None for k in self.output}
            del params
            if self._mm_models:
                for model_id in self._mm_models:
                    memory_manager.remove(model_id)
                self._mm_models = []

            try:
                output = getattr(self, self.CALLBACK)(**self.params)
            except Exception as e:
                self.params = {}
                #self.output = {k: None for k in self.output}
                raise RuntimeError(f"Error executing {self.module_name}.{self.class_name}: {e}")

            if isinstance(output, dict):
                # output and self.output keys must be the same
                if set(output.keys()) != set(self.output.keys()):
                    raise ValueError(f"Module {self.module_name}.{self.class_name}: Output keys do not match: {output.keys()} != {self.output.keys()}")

                self.output = output
            else:
                if len(self.output) > 1:
                    raise ValueError(f"Module {self.module_name}.{self.class_name}: Only one output returned, but multiple are expected ({self.output.keys()})")
                # if only one output is returned, assign it to the first output
                self.output[next(iter(self.output))] = output

        memory_flush()
        return self.output
    
    def __del__(self):
        try:
            if sys.meta_path is None:
                return  # Python is shutting down, skip cleanup
            
            for model_id in self._mm_models:
                memory_manager.remove(model_id)
        except (ImportError, AttributeError, RuntimeError):
            # Python is shutting down or import system is unavailable
            pass
        
        del self.params, self.output
    
    def pipe_callback(self, pipe, step_index, timestep, callback_kwargs):
        if not self.node_id:
            return
        
        if self._interrupt:
            pipe._interrupt = True

        if hasattr(pipe, '_cfg_cutoff_step') and pipe._cfg_cutoff_step is not None:
            cutoff_step = int(pipe._num_timesteps * pipe._cfg_cutoff_step)
            if step_index == cutoff_step:
                pipe._guidance_scale = 0.0
                callback_kwargs['prompt_embeds'] = callback_kwargs['prompt_embeds'][-1:]
                callback_kwargs['pooled_prompt_embeds'] = callback_kwargs['pooled_prompt_embeds'][-1:]
        
        progress = int((step_index + 1) / pipe._num_timesteps * 100)
        self.progress(progress)
        
        return callback_kwargs


    """
    ╭───────────╮
      WebSocket
    ╰───────────╯
    """

    def ws_message(self, message):
        if not self._sid:
            return
        
        server.queue_message(message, self._sid)

    def progress(self, progress: int):
        if not self._sid or not self.node_id:
            return
                
        server.queue_message({
            "type": "progress",
            "node": self.node_id,
            "progress": progress,
        }, self._sid)


    """
    ╭────────────────╮
      Memory Manager
    ╰────────────────╯
    """

    def mm_add(self, model, priority=1):
        if self.node_id is None:
            return model

        model_id = memory_manager.add(model, priority)
        
        if model_id not in self._mm_models:
            self._mm_models.append(model_id)
        
        return model_id
    
    def mm_remove(self, model):
        if self.node_id is None:
            return model.to('cpu')

        model_id = memory_manager.remove(model)
        if model_id is not None:
            self._mm_models.remove(model_id)
        
        return model_id
    
    def mm_get(self, model):
        if self.node_id is None:
            return model
        
        return memory_manager.get_model(model)
    
    def mm_update(self, model, **kwargs):       
        if self.node_id is None:
            return
        
        return memory_manager.update(model, **kwargs)
    
    def mm_load(self, model, device=None):
        if self.node_id is None:
            return model.to(device)
        
        return memory_manager.load_model(model, device)
    
    def mm_exec(self, func, device, exclude=[], args=None, kwargs=None):
        if self.node_id is None:
            return func(*args, **kwargs)
        
        return memory_manager.exec(func, device, exclude, args, kwargs)
    
    def mm_unload_all(self, device=None):
        if self.node_id is None:
            return
        
        memory_manager.unload_all(device)
