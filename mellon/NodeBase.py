import logging
logger = logging.getLogger('mellon')
from modules import MODULE_MAP
from mellon.server import server
from mellon.config import CONFIG
from mellon.modelstore import modelstore
from utils.memory_menager import memory_manager
import numpy as np
import torch
import sys
from huggingface_hub.utils import LocalEntryNotFoundError

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

    # For PIL-like images
    if hasattr(a, 'tobytes') and callable(a.tobytes):
        if a.size != b.size or a.mode != b.mode:
            return False
        return a.tobytes() == b.tobytes()

    # For numpy arrays
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        return np.array_equal(a, b)

    # For lists and tuples, check length and then each element recursively
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))

    # For sets, the standard equality check should be sufficient
    if isinstance(a, set):
        return a == b

    # For dictionaries, check keys and then each value recursively
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[key], b[key]) for key in a)

    # 4. Generic object checks
    if hasattr(a, 'to_dict') and callable(a.to_dict):
        return deep_equal(a.to_dict(), b.to_dict())

    # As a fallback, compare the objects' __dict__ attributes recursively
    if hasattr(a, '__dict__'):
        return deep_equal(a.__dict__, b.__dict__)

    # 5. Default case: use standard equality for primitives (int, str, etc.)
    return a == b

def recursive_type_cast(value, ttype, key):
    if value is None:
        return None

    if isinstance(value, list):
        return [recursive_type_cast(v, ttype, key) for v in value]
    if isinstance(value, dict):
        return {k: recursive_type_cast(v, ttype, f"{key}.{k}") for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(recursive_type_cast(list(value), ttype, key))
        #return tuple(recursive_type_cast(v, type, f"{key}.{i}") for i, v in enumerate(value))
    if isinstance(value, np.ndarray):
        return np.array(recursive_type_cast(value.tolist(), ttype, key), dtype=value.dtype)
    
    try:       
        if ttype.startswith('int'):
            if isinstance(value, str) and value.strip() == '':
                return 0
            return int(float(value))  # Convert via float first to handle "1.0" -> 1
        if ttype.startswith('float'):
            if isinstance(value, str) and value.strip() == '':
                return 0.0
            return float(value)
        if ttype.startswith('str') or ttype.startswith('text'):
            return str(value or '')
        if ttype.startswith('bool'):
            if isinstance(value, str):
                # Handle string representations of booleans
                value_lower = value.lower().strip()
                if value_lower in ('true', '1', 'yes', 'on', 'y'):
                    return True
                if value_lower in ('false', '0', 'no', 'off', 'n'):
                    return False
                try:
                    return bool(float(value_lower))
                except ValueError:
                    return value
            return bool(value)
        return value
    except Exception:
        return value

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
        self._has_changed = False
        self._execution_time = { 'last': None, 'min': None, 'max': None }
        self._memory_usage = { 'last': None, 'min': None, 'max': None }
        self._mm_models = []
        self._interrupt = False
        self._skip_params_check = MODULE_MAP[self.module_name][self.class_name].get('skipParamsCheck', False)

    def __call__(self, **kwargs):
        self._interrupt = False
        
        if self._skip_params_check:
            params = kwargs
        else:
            # filter out params that are not in the default_params
            params = { key: kwargs[key] for key in kwargs if key in self.default_params }

        # if node_id is None, the class was called directly, so we execute it without further processing
        if self.node_id is None:
            return getattr(self, self.CALLBACK)(**params)
        
        # params normalization and validation
        if not self._skip_params_check:
            for key, value in params.items():
                if value is None:
                    value = self.default_params[key]['default'] if 'default' in self.default_params[key] else None
                    params[key] = value
                
                if 'type' in self.default_params[key]:
                    type = self.default_params[key]['type']
                    if isinstance(type, list):
                        type = type[0]
                    params[key] = recursive_type_cast(value, type, key)
                
                if 'options' in self.default_params[key] and not self.default_params[key].get('fieldOptions', {}).get('noValidation', False):
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
        
        # if we added a model not in the huggingface cache, we set a flag that
        # later will be used to tell the client to update its local cache
        update_hf_cache = False
        update_local_cache = False
        for key in self.default_params:
            is_modelselect = self.default_params[key].get('display') == 'modelselect'
            if is_modelselect:
                if isinstance(params[key], str):
                    sources = self.default_params[key].get('fieldOptions', { 'sources': ['hub'] }).get('sources', ['hub'])
                    params[key] = { 'source': sources[0], 'value': params[key] }

                if params[key].get('source') == 'hub':
                    if not modelstore.is_hf_cached(params[key].get('value')):
                        update_hf_cache = True
                elif params[key].get('source') == 'local':
                    if not modelstore.is_local_cached(params[key].get('value')):
                        update_local_cache = True

        # post processing
        for key in self.default_params:
            if 'postProcess' in self.default_params[key]:
                # we pass the current value and the dict of all the values for cross parameter validation
                params[key] = self.default_params[key]['postProcess'](params[key], params)
        
        self._has_changed = False # flag to know if the node has changed since the last execution

        # if any of the values has changed or self.output is empty, we need to execute the node
        if (not deep_equal(self.params, params)) or any(v is None for v in self.output.values()):
            self._has_changed = True
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

            if output and isinstance(output, dict):
                # output and self.output keys must be the same
                if set(output.keys()) != set(self.output.keys()) and not self._skip_params_check:
                    raise ValueError(f"Module {self.module_name}.{self.class_name}: Output keys do not match: {output.keys()} != {self.output.keys()}")

                self.output = output
            # elif output is not None:
            #     if len(self.output) > 1:
            #         raise ValueError(f"Module {self.module_name}.{self.class_name}: Only one output returned, but multiple are expected ({self.output.keys()})")
            #     # if only one output is returned, assign it to the first output
            #     self.output[next(iter(self.output))] = output
            
            # inform the client that the huggingface cache needs to be updated
            if update_hf_cache:
                modelstore.update_hf()
                server.queue_message({
                    "type": "hf_cache_update",
                    "node": self.node_id,
                }, self._sid)
            if update_local_cache:
                modelstore.update_local()
                server.queue_message({
                    "type": "local_cache_update",
                    "node": self.node_id,
                }, self._sid)

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
    
    def graceful_model_loader(self, callback, model_id, config, local_files_only=True):
        output = None
        online_status = CONFIG.hf['online_status']
        if online_status == 'Online':
            local_files_only = False
        
        if hasattr(callback, 'from_pretrained'):
            callback = callback.from_pretrained

        try:
            if model_id is None:
                output = callback(**config, local_files_only=local_files_only)
            else:
                output = callback(model_id, **config, local_files_only=local_files_only)

        except (LocalEntryNotFoundError, OSError) as e:
            if not local_files_only:
                raise e

            if online_status == 'Offline':
                logger.error(f"Model {model_id} is not available in offline mode. Consider changing online_status to 'Auto' or 'Online' in the config.ini file.")
                raise

            logger.warning(f"Model {model_id} not found locally, attempting to download...")
            output = self.graceful_model_loader(callback, model_id, config, local_files_only=False)
            modelstore.update_hf()
        except Exception as e:
            logger.error(f"Error loading {model_id}: {e}")
            raise
        
        return output
    
    def pipe_callback(self, pipe, step_index, timestep, callback_kwargs):
        if not self.node_id:
            return
        
        if self._interrupt:
            pipe._interrupt = True

        if hasattr(pipe, '_cfg_cutoff_step') and pipe._cfg_cutoff_step is not None:
            cutoff_step = int(pipe._num_timesteps * pipe._cfg_cutoff_step)
            if step_index == cutoff_step:
                pipe._guidance_scale = 0.0
                if 'prompt_embeds' in callback_kwargs:
                    callback_kwargs['prompt_embeds'] = callback_kwargs['prompt_embeds'][-1:]
                if 'pooled_prompt_embeds' in callback_kwargs:
                    callback_kwargs['pooled_prompt_embeds'] = callback_kwargs['pooled_prompt_embeds'][-1:]
        
        progress = int((step_index + 1) / pipe._num_timesteps * 100)
        self.progress(progress)
        
        return callback_kwargs
    
    def trigger_output(self, output, value=None):
        if not self.node_id or output not in self.output:
            return
        
        if value is not None:
            self.output[output] = value

        server.trigger_node(self.node_id, output, self._sid)


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

    def send_node_definition(self, params):
        if not self._sid or not self.node_id:
            return

        server.queue_message({
            "type": "node_definition",
            "node": self.node_id,
            "params": params,
        }, self._sid)

    def set_field_visibility(self, fields: dict):
        if not self._sid or not self.node_id:
            return
        
        server.queue_message({
            "type": "set_field_visibility",
            "node": self.node_id,
            "fields": fields,
        }, self._sid)
    
    def set_field_value(self, field: dict):
        if not self._sid or not self.node_id:
            return
        
        server.queue_message({
            "type": "set_field_value",
            "node": self.node_id,
            "fields": field,
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
    
    def mm_exec(self, func, device, models=[], exclude=[], args=None, kwargs=None):
        if self.node_id is None:
            return func(*args, **kwargs)
        
        return memory_manager.exec(func, device, models, exclude, args, kwargs)
    
    def mm_unload_all(self, device=None):
        if self.node_id is None:
            return
        
        memory_manager.unload_all(device)
