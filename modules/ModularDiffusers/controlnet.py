import importlib
import logging

from mellon.NodeBase import NodeBase

from . import components
from .utils import combine_multi_inputs, collect_model_ids
from .modular_utils import pipeline_class_to_mellon_node_config


logger = logging.getLogger("mellon")




class MultiControlNet(NodeBase):
    label = "Multi ControlNet"
    category = "adapters"

    params = {
        "controlnet_list": {
            "label": "ControlNet",
            "display": "input",
            "type": "custom_controlnet",
            "spawn": True,
        },
        "controlnet": {
            "label": "Multi Controlnet",
            "type": "custom_controlnet",
            "display": "output",
        },
    }

    def execute(self, controlnet_list):
        controlnet = combine_multi_inputs(controlnet_list)
        return {"controlnet": controlnet}


class ControlnetUnion(NodeBase):
    label = "ControlNet Union"
    category = "adapters"

    params = {
        "pose_image": {
            "label": "Pose image",
            "type": "image",
            "display": "input",
        },
        "depth_image": {
            "label": "Depth image",
            "type": "image",
            "display": "input",
        },
        "edges_image": {
            "label": "Edges image",
            "type": "image",
            "display": "input",
        },
        "lines_image": {
            "label": "Lines image",
            "type": "image",
            "display": "input",
        },
        "normal_image": {
            "label": "Normal image",
            "type": "image",
            "display": "input",
        },
        "segment_image": {
            "label": "Segment image",
            "type": "image",
            "display": "input",
        },
        "tile_image": {
            "label": "Tile image",
            "type": "image",
            "display": "input",
        },
        "repaint_image": {
            "label": "Repaint image",
            "type": "image",
            "display": "input",
        },
        "controlnet_conditioning_scale": {
            "label": "Scale",
            "type": "float",
            "display": "slider",
            "default": 0.5,
            "min": 0,
            "max": 1,
        },
        "control_guidance_start": {
            "label": "Start",
            "type": "float",
            "display": "slider",
            "default": 0.0,
            "min": 0,
            "max": 1,
        },
        "control_guidance_end": {
            "label": "End",
            "type": "float",
            "display": "slider",
            "default": 1.0,
            "min": 0,
            "max": 1,
        },
        "controlnet_model": {
            "label": "Controlnet Union Model",
            "type": "diffusers_auto_model",
            "display": "input",
        },
        "controlnet": {
            "label": "Controlnet",
            "display": "output",
            "type": "custom_controlnet",
        },
    }

    def execute(
        self,
        pose_image,
        depth_image,
        edges_image,
        lines_image,
        normal_image,
        segment_image,
        tile_image,
        repaint_image,
        controlnet_conditioning_scale,
        controlnet_model,
        control_guidance_start,
        control_guidance_end,
    ):
        image_map = {
            "pose_image": (pose_image, 0),
            "depth_image": (depth_image, 1),
            "edges_image": (edges_image, 2),
            "lines_image": (lines_image, 3),
            "normal_image": (normal_image, 4),
            "segment_image": (segment_image, 5),
            "tile_image": (tile_image, 6),
            "repaint_image": (repaint_image, 7),
        }

        control_mode = []
        control_image = []

        for key, (image, index) in image_map.items():
            if image is not None:
                control_mode.append(index)
                control_image.append(image)

        controlnet = {
            "controlnet_model": controlnet_model,
            "controlnet_inputs": {
                "control_image": control_image,
                "control_mode": control_mode,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            },
        }
        return {"controlnet": controlnet}


class Controlnet(NodeBase):
    label = "ControlNet"
    category = "adapters"
    resizable = True
    skipParamsCheck = True
    node_type = "controlnet"

    params = {
        "model_type": {"label": "Model Type", "type": "string", "default": "", "hidden": True},
        "controlnet_bundle": {
            "label": "Controlnet",
            "display": "output",
            "type": "custom_controlnet",
            "onSignal": [
                {
                    "action": "value",
                    "target": "model_type",
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
            ],
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def update_node(self, values, ref):

        node_params = {
            "model_type": {
                "label": "Model Type",
                "type": "string",
                "default": "",
                "hidden": True,
            },
            "controlnet_bundle": {
                "label": "Controlnet",
                "display": "output",
                "type": "custom_controlnet",
                "onSignal": [
                    {
                        "action": "value",
                        "target": "model_type",
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
                ],
            },
        }

        model_type = values.get("model_type", "")

        if model_type == "" or self._model_type == model_type:
            return None

        self._model_type = model_type

        diffusers_module = importlib.import_module("diffusers")
        self._pipeline_class = getattr(diffusers_module, model_type)
        
        _, node_config = pipeline_class_to_mellon_node_config(self._pipeline_class, self.node_type)

        # Not supported for this pipeline
        if node_config is None:
            self.send_node_definition(node_params)
            return

        node_params_to_update = node_config.to_mellon_dict()["params"]
        node_params_to_update.pop("controlnet_bundle", None)

        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def execute(self, **kwargs):
        kwargs = dict(kwargs)

        # 1. Get node config
        blocks, node_config = pipeline_class_to_mellon_node_config(
            self._pipeline_class, self.node_type
        )
        denoise_blocks, _ = pipeline_class_to_mellon_node_config(self._pipeline_class, "denoise")
        if denoise_blocks is None:
            return

        # 2. Cast params to correct types (Mellon bug workaround)
        # YiYi notes: should fix and remove in the future
        for param_name, param_config in node_config.inputs.items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        node_output = None
        if blocks is not None:
            # 3. Create pipeline
            self._pipeline = blocks.init_pipeline(components_manager=components)


            # 4. Update components
            expected_component_names = blocks.component_names
            model_input_names = list(node_config.model_inputs.keys()) if node_config.model_inputs else []
            model_ids = collect_model_ids(
                kwargs,
                target_key_names=model_input_names,
                target_model_names=expected_component_names,
            )

            if model_ids:
                components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
                if components_to_update:
                    self._pipeline.update_components(**components_to_update)

            # 5. Compile runtime inputs from kwargs based on node_config.inputs
            node_kwargs = {}
            input_names = list(node_config.inputs.keys()) if node_config.inputs else []

            for name in input_names:
                if name not in kwargs:
                    continue
                value = kwargs.get(name)

                if isinstance(value, dict) and name not in blocks.input_names:
                    for k, v in value.items():
                        if k in blocks.input_names:
                            node_kwargs[k] = v
                elif name in blocks.input_names:
                    node_kwargs[name] = value

            # 6. Run the pipeline
            node_output = self._pipeline(**node_kwargs).values

        # 7. Prepare controlnet output for Denoise node
        # use the denoise blocks to know what inputs it expects

        controlnet_inputs = {}
        for name in denoise_blocks.input_names:
            if node_output and name in node_output and node_output[name] is not None:
                controlnet_inputs.update({name: node_output[name]})
            elif name in kwargs and kwargs[name] is not None:
                controlnet_inputs.update({name: kwargs.pop(name)})

        # YiYi TODO: list controlnet as required/static model input for controlnet node
        controlnet_out = {
            "controlnet": kwargs.get("controlnet"),
            **controlnet_inputs,
        }

        return {"controlnet_bundle": controlnet_out}
