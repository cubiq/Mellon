from mellon.NodeBase import NodeBase

from .utils import combine_multi_inputs
from diffusers import ModularPipeline
from . import components

import logging
logger = logging.getLogger("mellon")
logger.setLevel(logging.DEBUG)

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
            "type": "controlnet",
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
            "type": "controlnet",
            "spawn": True,
        },
        "controlnet": {
            "label": "Multi Controlnet",
            "type": "controlnet",
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
            "type": "controlnet",
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



class DynamicControlnet(NodeBase):
    label = "Dynamic ControlNet"
    category = "adapters"
    resizable = True
    skipParamsCheck = True
    # YiYi Notes: in the future, we should support any repo_id without the option list
    params = {
        "repo_id": {
            "label": "Repository ID",
            "type": "string",
            "options": {
                "": "",
                "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
                "Qwen/Qwen-Image": "qwenimage",
                "Qwen/Qwen-Image-Edit": "qwenimage-edit",
            },
            "value": "stabilityai/stable-diffusion-xl-base-1.0",
            "onChange": "updateNode",
            },
        }

    # Keep a common schema for all controlnet inputs/outputs
    @property
    def all_params_list(self):
        return {
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
            "width": {"label": "Width", "type": "int", "default": 1024, "min": 64, "step": 8},
            "height": {"label": "Height", "type": "int", "default": 1024, "min": 64, "step": 8},
        }

    @property
    def all_models_list(self):
        return {
            "controlnet": {
                "label": "Controlnet Model",
                "type": "diffusers_auto_model",
                "display": "input",
            },
            "vae": {
                "label": "VAE", 
                "display": "input", 
                "type": "diffusers_auto_model",
                "onChange": {False: [], True: ["width", "height"]},
            },
        }

    
    # YiYi Notes: this does not work very well
    # e.g. if I have already created a dynamic controlnet node with qwenimage; 
    # everytime I restart the app, the node UI stays as a qwenimage controlnet node, 
    # but here the self.cobtrol_node is None,ideally we should have a way to sync the node with the value in UI
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.blocks = None
        self._controlnet_node = None

    def updateNode(self, values, ref):

        # default params
        controlnet_params = {
            "repo_id": {
                "label": "Repository ID",
                "type": "string",
                "options": {
                    "": "",
                    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
                    "Qwen/Qwen-Image": "qwenimage",
                    "Qwen/Qwen-Image-Edit": "qwenimage-edit",
                },
                "value": "stabilityai/stable-diffusion-xl-base-1.0",
                "onChange": "updateNode",
                },
        }     

        repo_id = values.get("repo_id", "")
        try:
            self.blocks = ModularPipeline.from_pretrained(repo_id).blocks
        except Exception as e:
            logger.debug(f" Faled to load the node from {repo_id}: {e}")
            self.blocks = None
        
        is_support_controlnet = False
        if self.blocks is not None and "control_image" in self.blocks.input_names:
            is_support_controlnet = True

        if not is_support_controlnet:
            self.send_node_definition(controlnet_params)
            return

        all_inputs = self.blocks.input_names

        # check if there is an standalone block for controlnet, e.g. vae encoder
        # YiYi TODO: very hacky code here, refactor and standardize how blocks are structured in modular diffusers source code
        controlnet_pre_block = None
        block_names = [name for name in self.blocks.sub_blocks.keys()]
        for name in block_names:
            if "controlnet" in name.lower():
                if controlnet_pre_block is not None:
                    logger.debug(f" Found multiple controlnet blocks: {name}")
                controlnet_pre_block = self.blocks.sub_blocks.pop(name)
            elif "input" != name:
                self.blocks.sub_blocks.pop(name)
            elif "input" == name:
                break
        self.blocks.sub_blocks.pop("decode")

        # if we support controlnet, a controlnet model and a control image are required
        controlnet_params.update({
            "controlnet": {
                "label": "Controlnet Model",
                "type": "diffusers_auto_model",
                "display": "input",
            },
            "control_image": {
                "label": "Control Image",
                "type": "image",
                "display": "input",
            },
            "controlnet_out": {
                "label": "Controlnet",
                "display": "output",
                "type": "controlnet",
                },
        })
        if controlnet_pre_block is not None:
            self._controlnet_node = controlnet_pre_block.init_pipeline(repo_id, components_manager=components)
            pre_components = self._controlnet_node.pretrained_component_names
        else:
            self._controlnet_node = None
            pre_components = []

        # adding additional controlnet params the blocks accepts
        for input_param in self.all_params_list:
            if input_param in all_inputs and input_param not in controlnet_params:
                controlnet_params[input_param] = self.all_params_list[input_param]

        # adding adiditional model components needed to run in ths prepare controlnet block (in addition to the controlnet model)
        for comp_name in pre_components:
            if comp_name in self.all_models_list and comp_name not in controlnet_params:
                controlnet_params[comp_name] = self.all_models_list[comp_name]

        self.send_node_definition(controlnet_params)

    def execute(self, **kwargs):

        if self.blocks is None:
            return

        if self._controlnet_node is not None:
            pre_comp_dict = {}
            pre_input_dict = {}
            for comp_name in self._controlnet_node.pretrained_component_names:
                if comp_name in kwargs:
                    comp_dict = kwargs[comp_name]
                    if comp_name == "controlnet":
                        controlnet = comp_dict
                    pre_comp_dict.update(components.get_components_by_ids([comp_dict["model_id"]]))
            self._controlnet_node.update_components(**pre_comp_dict)
            for input_name in self._controlnet_node.blocks.input_names:
                if input_name in kwargs:
                    input_value = kwargs.pop(input_name)
                    pre_input_dict.update({input_name: input_value})

            control_out = self._controlnet_node(**pre_input_dict)
            kwargs.update(control_out.values)
        
        controlnet_inputs_dict = {}
        for name in self.blocks.input_names:
            if name in kwargs and kwargs[name] is not None:
                controlnet_inputs_dict.update({name: kwargs.pop(name)})

        controlnet = {
            "controlnet_model": controlnet,
            "controlnet_inputs": controlnet_inputs_dict,
        }

        logger.debug(f" controlnet output: {controlnet}")
        
        return {"controlnet_out": controlnet}
