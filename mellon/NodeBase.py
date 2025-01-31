import logging
logger = logging.getLogger('mellon')
from modules import MODULE_MAP
import torch
import time
from utils.memory_manager import memory_flush, memory_manager
from mellon.server import web_server
import nanoid
import numpy as np

def get_module_params(module_name, class_name):
    params = MODULE_MAP[module_name][class_name]['params'] if module_name in MODULE_MAP and class_name in MODULE_MAP[module_name] else {}
    return { p: params[p]['default'] if 'default' in params[p] else None
            for p in params if not 'display' in params[p] or (params[p]['display'] != 'output' and params[p]['display'] != 'ui') }

def get_module_output(module_name, class_name):
    params = MODULE_MAP[module_name][class_name]['params'] if module_name in MODULE_MAP and class_name in MODULE_MAP[module_name] else {}
    return { p: None for p in params if 'display' in params[p] and params[p]['display'] == 'output' }

def filter_params(params, args):
    return { key: args[key] for key in args if key in params }

def has_changed(params, args):
    return any(params.get(key) != args.get(key) for key in args if key in params)

def are_different(a, b):
    # quick identity check
    if a is b:
        return False

    # check if the types are different
    if type(a) != type(b):
        return True

    # check custom hash, this value is king
    if hasattr(a, "_MELLON_HASH") and hasattr(b, "_MELLON_HASH"):
       return hasattr(a, "_MELLON_HASH") != hasattr(b, "_MELLON_HASH")
    
    # common attributes
    if hasattr(a, 'shape'):
        if a.shape != b.shape:
            return True
    if hasattr(a, 'dtype'):
        if not hasattr(b, 'dtype') or a.dtype != b.dtype:
            return True

    # quick image comparison
    if hasattr(a, 'size'):
        if a.size != b.size:
            return True
    if hasattr(a, 'mode'):
        if a.mode != b.mode:
            return True

    # deep PIL images comparison
    if hasattr(a, 'getdata') and hasattr(a, 'width') and hasattr(a, 'height'):
        # compare small images with tobytes(), possibly unnecessary optimization
        if a.width*a.height < 32768:
            return a.tobytes() != b.tobytes()
        return not np.array_equal(np.asarray(a), np.asarray(b))

    # trimesh comparison
    if hasattr(a, 'vertices') and hasattr(a.vertices, 'shape'):
        if are_different(a.vertices, b.vertices):
            return True
        if hasattr(a, 'visual') and hasattr(a.visual, 'material') and hasattr(a.visual.material, 'image'):
            if are_different(a.visual.material.image, b.visual.material.image):
                return True
        if hasattr(a, 'faces') and hasattr(a.faces, 'shape'):
            if are_different(a.faces, b.faces):
                return True
        # we assume that the mesh is the same if the vertices, faces and material are the same
        # TODO: check if this is correct
        return False

    # compare numpy arrays
    if isinstance(a, np.ndarray):
        return not np.array_equal(a, b)

    # compare tensors
    if isinstance(a, torch.Tensor):
        return not torch.equal(a, b)

    # iterate list, tuple, dict
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return True
        return any(are_different(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return True
        return any(are_different(a[k], b[k]) for k in a)

    if hasattr(a, 'to_dict'):
        x = a.to_dict()
        y = b.to_dict()
        return any(are_different(x[k], y[k]) for k in x)

    if hasattr(a, '__dict__') and hasattr(b, '__dict__'):
        if a.__dict__ != b.__dict__:
            return True

    if a != b:
        return True

    return False

class NodeBase():
    CALLBACK = 'execute'
    FORCE_UNLOAD = True

    def __init__(self, node_id=None):
        self.node_id = node_id
        self.module_name = self.__class__.__module__.split('.')[-1]
        if 'custom.' in self.__class__.__module__:
            self.module_name = self.module_name + '.custom'
        self.class_name = self.__class__.__name__
        self.params = {}
        self.output = get_module_output(self.module_name, self.class_name)
        
        self._client_id = None
        self._pipe_interrupt = False
        self._mm_model_ids = []
        self._execution_time = 0

    def __call__(self, **kwargs):
        self._pipe_interrupt = False

        # if the node_id is not set, the class was called by the user and it's not part of a workflow,
        # we execute the method directly
        if not self.node_id:
            params = { key: kwargs[key] for key in kwargs if not key.startswith('__') }
            return getattr(self, self.CALLBACK)(**params)

        values = self._validate_params(kwargs)

        execution_time = time.time()

        if self._has_changed(values) or self._is_output_empty():
            self.params.update(values)

            # delete previously loaded models
            # TODO: delete a model only if something changed about it
            if self._mm_model_ids:
                memory_manager.delete_model(self._mm_model_ids, unload=self.FORCE_UNLOAD)
                self._mm_model_ids = []

            try:
                params = { key: self.params[key] for key in self.params if not key.startswith('__') }
                output = getattr(self, self.CALLBACK)(**params)
            except Exception as e:
                self.params = {}
                self.output = get_module_output(self.module_name, self.class_name)
                memory_flush(gc_collect=True)
                raise e

            if isinstance(output, dict):
                # Overwrite output values only for existing keys
                #self.output.update({k: output[k] for k in self.output if k in output})
                self.output = output
            else:
                # If only a single value is returned, assign it to the first output
                first_key = next(iter(self.output))
                self.output[first_key] = output

        self._execution_time = time.time() - execution_time

        # for good measure, flush the memory
        memory_flush()

        return self.output

    def __del__(self):
        del self.params, self.output # TODO: check if this actually works with cuda

        if self._mm_model_ids:
            memory_manager.delete_model(self._mm_model_ids, unload=self.FORCE_UNLOAD)

        memory_flush(gc_collect=True)

    def _validate_params(self, values):
        # get the parameters schema for the module/class
        schema = MODULE_MAP[self.module_name][self.class_name]['params'] if self.module_name in MODULE_MAP and self.class_name in MODULE_MAP[self.module_name] else {}

        # get the default values for the parameters
        defaults = get_module_params(self.module_name, self.class_name)

        # filter out any input args that are not valid parameters and exclude the special fields starting with __
        values = { key: values[key] for key in values if key in defaults and not key.startswith('__') }

        # ensure the values are of the correct type
        for key in values:
            if 'type' in schema[key]:
                # type can be a list, used to allow multiple types with input handles (a helper for the UI)
                # the main type is the first one in the list
                type = (schema[key]['type'][0] if isinstance(schema[key]['type'], list) else schema[key]['type']).lower()

                if type.startswith('int'):
                    values[key] = int(values[key]) if not isinstance(values[key], list) else [int(v) for v in values[key]]
                elif type == 'float':
                    values[key] = float(values[key]) if not isinstance(values[key], list) else [float(v) for v in values[key]]
                elif type.startswith('bool'):
                    values[key] = bool(values[key]) if not isinstance(values[key], list) else [bool(v) for v in values[key]]
                elif type.startswith('str'):
                    values[key] = str(values[key] or '') if not isinstance(values[key], list) else [str(v or '') for v in values[key]]

        # we perform a second pass for cross parameter validation when calling the postProcess function
        for key in values:
            # ensure the value is a valid option (mostly for dropdowns)
            if 'options' in schema[key] and not ('no_validation' in schema[key] and schema[key]['no_validation']):
                options = schema[key]['options']

                # options can be in the format: [ 1, 2, 3 ] or { '1': { }, '2': { }, '3': { } }
                if isinstance(options, list):
                    val = [values[key]] if not isinstance(values[key], list) else values[key]
                    if any(v not in options for v in val):
                        raise ValueError(f"Invalid value for {key}: {values[key]}")
                elif isinstance(options, dict):
                    val = [values[key]] if not isinstance(values[key], list) else values[key]
                    if any(v not in options for v in val):
                        raise ValueError(f"Invalid value for {key}: {values[key]}")
                else:
                    raise ValueError(f"Invalid options for {key}: {options}")

            # call the postProcess function if present
            if 'postProcess' in schema[key]:
                # we pass the value and the entire dict for cross parameter validation
                values[key] = schema[key]['postProcess'](values[key], values)

        # update the default values with the validated values
        defaults.update(values)

        return defaults

    def _has_changed(self, values):
        return any(
            key not in self.params or
            are_different(self.params.get(key), values.get(key))
            for key in values
        )
    
    def _is_output_empty(self):
        return all(value is None for value in self.output.values())
    
    def pipe_callback(self, pipe, step_index, timestep, kwargs):
        import asyncio
        if self.node_id:
            try:
                progress = int((step_index + 1) / pipe._num_timesteps * 100)
                asyncio.run_coroutine_threadsafe(
                    web_server.client_queue.put({
                        "client_id": self._client_id,
                        "data": {
                            "type": "progress",
                            "nodeId": self.node_id,
                            "progress": progress
                        }
                    }), 
                    web_server.event_loop
                )
            except Exception as e:
                logger.warning(f"Error queuing progress update: {str(e)}")

            # interrupt callback
            if self._pipe_interrupt:
                pipe._interrupt = True

        return kwargs
    
    def mm_add(self, model, model_id=None, device=None, priority=2):
        # if the node_id is not set, the class was called directly and we skip the memory manager
        # it's up to the caller to manage the model
        if not self.node_id:
            return model

        if memory_manager.is_cached(model_id):
            self.mm_update(model_id, model=model, priority=priority)
            return model_id

        model_id = f'{self.node_id}.{model_id}' if model_id else f'{self.node_id}.{nanoid.generate(size=8)}'
        device = device if device else str(model.device)

        self._mm_model_ids.append(model_id)
        return memory_manager.add_model(model, model_id, device=device, priority=priority)

    def mm_get(self, model_id):
        model_id = model_id if isinstance(model_id, str) else model_id._mm_id if hasattr(model_id, '_mm_id') else None
        return memory_manager.get_model(model_id) if model_id else None

    def mm_load(self, model_id, device):
        model_id = model_id if isinstance(model_id, str) else model_id._mm_id if hasattr(model_id, '_mm_id') else None
        return memory_manager.load_model(model_id, device) if model_id else None

    def mm_unload(self, model_id):
        model_id = model_id if isinstance(model_id, str) else model_id._mm_id if hasattr(model_id, '_mm_id') else None
        return memory_manager.unload_model(model_id) if model_id else None

    def mm_update(self, model_id, model=None, priority=None, unload=True):
        model_id = model_id if isinstance(model_id, str) else model_id._mm_id if hasattr(model_id, '_mm_id') else None
        return memory_manager.update_model(model_id, model=model, priority=priority, unload=unload) if model_id else None
    
    def mm_inference(self, func, device, exclude=None, no_grad=False):
        exclude_list = []
        if exclude:
            exclude = [exclude] if not isinstance(exclude, list) else exclude
            for model in exclude:
                if isinstance(model, str):
                    exclude_list.append(model)
                elif hasattr(model, '_mm_id'):
                    exclude_list.append(model._mm_id)

        while True:
            try:
                with torch.inference_mode() if not no_grad else torch.no_grad():
                    return func()
            except torch.OutOfMemoryError as e:
                if memory_manager.unload_next(device, exclude=exclude):
                    continue
                else:
                    raise e
    
    def mm_flash_load(self, model, model_id=None, device='cpu', priority=3):
        model_id = f'{self.node_id}.{model_id}' if model_id else f'{self.node_id}.{nanoid.generate(size=8)}'
        device = device if device else str(model.device)

        return memory_manager.flash_load(model, model_id, device=device, priority=priority)
