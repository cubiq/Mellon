import logging

from diffusers.modular_pipelines import ModularPipeline
import importlib

from mellon.NodeBase import NodeBase
from .utils import collect_model_ids
from .modular_utils import pipeline_class_to_mellon_node_config

from . import components


logger = logging.getLogger("mellon")

class EncodePrompt(NodeBase):
    label = "Encode Prompt"
    category = "embedding"
    resizable = True
    skipParamsCheck = True
    node_type = "text_encoder"
    params = {
        "model_type": {
            "label": "Model Type", 
            "type": "string", 
            "default": "", 
            "hidden": True  # Hidden field to receive signal data
        },
        "text_encoders": {
            "label": "Text Encoders *",
            "type": "diffusers_auto_models",
            "display": "input",
            "onSignal": [
                {
                    "action": "value",
                    "target": "model_type",
                    # "data": SIGNAL_DATA, # YiYi Notes: not working
                    "data": {
                        "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
                        "QwenImageModularPipeline": "QwenImageModularPipeline",
                        "QwenImageEditModularPipeline": "QwenImageEditModularPipeline",
                        "QwenImageEditPlusModularPipeline": "QwenImageEditPlusModularPipeline",
                        "FluxModularPipeline": "FluxModularPipeline",
                        "FluxKontextModularPipeline": "FluxKontextModularPipeline",
                    },
                },
                {"action": "exec", "data": "update_node"},
            ]
        },
    }

    def update_node(self, values, ref):

        node_params  = {
            "model_type": {
                "label": "Model Type", 
                "type": "string", 
                "default": "", 
                "hidden": True  # Hidden field to receive signal data
            },
            "text_encoders": {
                "label": "Text Encoders *",
                "display": "input",
                "type": "diffusers_auto_models",
                "onSignal": [
                    {
                        "action": "value",
                        "target": "model_type",
                        # "data": SIGNAL_DATA, # YiYi Notes: not working
                        "data": {
                            "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
                            "QwenImageModularPipeline": "QwenImageModularPipeline",
                            "QwenImageEditModularPipeline": "QwenImageEditModularPipeline",
                            "QwenImageEditPlusModularPipeline": "QwenImageEditPlusModularPipeline",
                            "FluxModularPipeline": "FluxModularPipeline",
                            "FluxKontextModularPipeline": "FluxKontextModularPipeline",
                        },
                    },
                    {"action": "exec", "data": "update_node"},
                ]
            },
        }
        model_type = values.get("model_type", "")

        if model_type == "" or self._model_type == model_type:
            return None

        self._model_type = model_type

        diffusers_module = importlib.import_module("diffusers")
        self._pipeline_class = getattr(diffusers_module, model_type)

        _, node_config = pipeline_class_to_mellon_node_config(self._pipeline_class, self.node_type)
        # not support this node type
        if node_config is None:
            self.send_node_definition(node_params)
            return

        node_params_to_update = node_config["params"]
        node_params_to_update.pop("text_encoders", None)
        
        node_params.update(**node_params_to_update)
        # YiYi TODO: can we perserve the current user values in the UI for "string"/"float"/"int" params?
        self.send_node_definition(node_params)

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def execute(self, **kwargs):

        kwargs = dict(kwargs)
        # 1. Get node config
        blocks, node_config = pipeline_class_to_mellon_node_config(
            self._pipeline_class, self.node_type
        )

        # 2. create pipeline
        repo_id = kwargs.get("text_encoders")["repo_id"]
        self._pipeline = blocks.init_pipeline(repo_id, components_manager=components)

        # YiYi Notes: take an extra step to cast the params to the correct type. 
        # This due to Mellon bugs, should not need to take this step.
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        # 3. update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        model_ids = collect_model_ids(
            kwargs, 
            target_key_names=model_input_names, 
            target_model_names=expected_component_names
        )
        
        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)

        # 4. compile a dict of runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]
        for name in input_names:
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            # if a dict is passed and is not an pipeline input, we unpack and process its contents
            # e.g. `embeddings` from text_encoder node
            if isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
                    else:
                        expected_inputs = "\n  - ".join(blocks.input_names)
                        logger.warning(
                            f"Input '{name}:{k}' is not expected by {self.node_type} blocks.\n"
                            f"Expected inputs:\n  - {expected_inputs} \n"
                            f"Blocks: {blocks}"
                            )
            # pass the value as it is to the pipeline
            elif name in blocks.input_names:
                node_kwargs[name] = value
            else:
                expected_inputs = "\n  - ".join(blocks.input_names)
                logger.warning(
                    f"Input '{name}' is not expected by {self.node_type} blocks.\n"
                    f"Expected inputs:\n  - {expected_inputs} \n"
                    f"Blocks: {blocks}"
                    )


        # 5. run the pipeline,
        node_output_state = self._pipeline(**node_kwargs)
        
        # 6. prepare the outputs dict based on node_config["output_names"]
        output_names = node_config["output_names"].copy()
        outputs = {}
        for name in output_names:
            if name == "doc":
                outputs["doc"] = self._pipeline.blocks.doc
            elif name == "embeddings":
                outputs["embeddings"] = node_output_state.get_by_kwargs("denoiser_input_fields")
            else:
                outputs[name] = node_output_state.get(name)
        return outputs
