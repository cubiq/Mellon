import importlib
import logging

from diffusers import ModularPipeline
from diffusers.modular_pipelines import SequentialPipelineBlocks

from mellon.NodeBase import NodeBase

from . import components
from .utils import combine_multi_inputs


logger = logging.getLogger("mellon")


class Controlnet(NodeBase):
    label = "ControlNet"
    category = "adapters"
    resizable = True

    params = {
        "control_image": {
            "label": "Control Image",
            "type": "image",
            "display": "input",
        },
        "controlnet_conditioning_scale": {
            "label": "Scale",
            "type": "float",
            "default": 0.5,
            "min": 0,
            "max": 1,
        },
        "control_guidance_start": {
            "label": "Start",
            "type": "float",
            "default": 0.0,
            "min": 0,
            "max": 1,
        },
        "control_guidance_end": {
            "label": "End",
            "type": "float",
            "default": 1.0,
            "min": 0,
            "max": 1,
        },
        "controlnet_model": {
            "label": "Controlnet Model",
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
        control_image,
        controlnet_conditioning_scale,
        controlnet_model,
        control_guidance_start,
        control_guidance_end,
    ):
        controlnet = {
            "controlnet_model": controlnet_model,
            "controlnet_inputs": {
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            },
        }
        return {"controlnet": controlnet}


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


def pipeline_class_to_mellon_node_config(pipeline_class, node_type=None):
    print(f" inside pipeline_class_to_mellon_node_config: {pipeline_class}")

    try:
        from diffusers.modular_pipelines.mellon_node_utils import ModularMellonNodeRegistry

        registry = ModularMellonNodeRegistry()
        node_type_config = registry.get(pipeline_class)[node_type]
    except Exception as e:
        logger.debug(f" Faled to load the node from {pipeline_class}: {e}")
        return None, None

    node_type_blocks = None
    pipeline = pipeline_class()

    if pipeline is not None and node_type_config is not None and node_type_config.blocks_names:
        blocks_dict = {
            name: block for name, block in pipeline.blocks.sub_blocks.items() if name in node_type_config.blocks_names
        }
        node_type_blocks = SequentialPipelineBlocks.from_blocks_dict(blocks_dict)

    return node_type_blocks, node_type_config


class DynamicControlnet(NodeBase):
    label = "Dynamic ControlNet"
    category = "adapters"
    resizable = True
    skipParamsCheck = True
    node_type = "controlnet"

    params = {
        "model_type": {"label": "Model Type", "type": "string", "default": "", "hidden": True},
        "controlnet_out": {
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
                    },
                },
                {"action": "exec", "data": "updateNode"},
            ],
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def updateNode(self, values, ref):
        # default params: repo_id
        controlnet_params = {}
        model_type = values.get("model_type", "")

        # skip import since we don't have a model type and diffusers import takes some time
        if model_type == "" or self._model_type == model_type:
            return None

        self._model_type = model_type

        diffusers_module = importlib.import_module("diffusers")
        self._pipeline_class = getattr(diffusers_module, model_type)

        _, controlnet_mellon_config = pipeline_class_to_mellon_node_config(self._pipeline_class, self.node_type)
        # not support this node type
        if controlnet_mellon_config is None:
            self.send_node_definition(controlnet_params)
            return

        # required params for controlnet
        controlnet_params.update(**controlnet_mellon_config.to_mellon_dict()["params"])

        self.send_node_definition(controlnet_params)

    def execute(self, **kwargs):
        controlnet = kwargs.get("controlnet", None)

        denoise_blocks, _ = pipeline_class_to_mellon_node_config(self._pipeline_class, "denoise")
        controlnet_blocks, _ = pipeline_class_to_mellon_node_config(self._pipeline_class, "controlnet")

        if denoise_blocks is None:
            return

        if controlnet_blocks is not None:
            # controlnet node
            controlnet_node = controlnet_blocks.init_pipeline(None, components_manager=components)

            # update the components for the controlnet node
            components_dict = {}
            for comp_name in controlnet_node.pretrained_component_names:
                if comp_name in kwargs:
                    model_info_dict = kwargs.pop(comp_name)
                    components_dict.update(components.get_components_by_ids([model_info_dict["model_id"]]))

            controlnet_node.update_components(**components_dict)

            inputs_dict = {}
            for input_name in controlnet_node.blocks.input_names:
                if input_name in kwargs and input_name not in inputs_dict:
                    inputs_dict[input_name] = kwargs.pop(input_name)

            controlnet_node_outputs = controlnet_node(**inputs_dict).values

        controlnet_out_inputs_dict = {}
        for name in denoise_blocks.input_names:
            if name in controlnet_node_outputs and controlnet_node_outputs[name] is not None:
                controlnet_out_inputs_dict.update({name: controlnet_node_outputs[name]})
            elif name in kwargs and kwargs[name] is not None:
                controlnet_out_inputs_dict.update({name: kwargs.pop(name)})

        controlnet = {
            "controlnet_model": controlnet,
            "controlnet_inputs": controlnet_out_inputs_dict,
        }

        logger.debug(f" controlnet output: {controlnet}")

        return {"controlnet_out": controlnet}
