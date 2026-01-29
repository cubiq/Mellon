import logging

from diffusers import ModularPipeline
from diffusers.modular_pipelines.mellon_node_utils import MellonPipelineConfig

from mellon.NodeBase import NodeBase

from . import components
from utils.torch_utils import str_to_dtype, DEFAULT_DEVICE, DEVICE_LIST


logger = logging.getLogger("mellon")
from . import components


class DynamicBlockNode(NodeBase):
    label = "Dynamic Block Node"
    resizable = True
    style = {"minWidth": 300}
    skipParamsCheck = True
    node_type = "custom"

    params = {
        "repo_id": {
            "label": "Custom Block",
            "display": "autocomplete",
            "type": "string",
            "default": "",
            "value": "",
            "options": {
                "": "",
                "OzzyGT/florence-2-block": "Florence-2 Block",
            },
            "fieldOptions": {"noValidation": True},
        },
        "load_block_button": {
            "label": "Load Custom Block",
            "display": "ui_button",
            "value": False,
            "onChange": "update_node",
        },
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "auto_offload": {"label": "Enable Auto Offload", "type": "boolean", "value": True},
        "doc": {
            "label": "Doc",
            "type": "string",
            "display": "output",
        },
    }

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        super().__del__()

    def _get_custom_params(self, repo_id):
        custom_mellon_config = MellonPipelineConfig.load(repo_id)
        custom_params = custom_mellon_config.node_params["custom"]["params"]

        return custom_params

    def update_node(self, values, ref):
        if not values.get("repo_id", ""):
            self.send_node_definition({})
            return

        repo_id = values.get("repo_id", "")
        custom_params = self._get_custom_params(repo_id)

        self.send_node_definition(custom_params)

    def execute(self, **kwargs):
        repo_id = kwargs.pop("repo_id", "")
        device = kwargs.pop("device", DEFAULT_DEVICE)
        auto_offload = kwargs.pop("auto_offload", True)
        
        pipeline = ModularPipeline.from_pretrained(
            repo_id, 
            trust_remote_code=True, 
            components_manager=components, 
            collection=self.node_id
        )

        # Load config to get input/output names and dtype
        custom_mellon_config = MellonPipelineConfig.load(repo_id)
        node_config = custom_mellon_config.node_params["custom"]
        
        # Get dtype from config
        default_dtype = custom_mellon_config.default_dtype
        if not default_dtype:
            default_dtype = "bfloat16"
        torch_dtype = str_to_dtype(default_dtype)
        
        # Enable auto offload if requested
        if auto_offload and (not components._auto_offload_enabled or components._auto_offload_device != device):
            components.enable_auto_cpu_offload(device=device)
        
        # Handle components
        components_update_dict = {}
        for comp_name in node_config.get("model_input_names", []):
            if comp_name in kwargs:
                model_info_dict = kwargs.pop(comp_name)
                components_update_dict.update(components.get_components_by_ids([model_info_dict["model_id"]]))


        # Check which components need to be loaded vs reused
        components_to_load = []
        for comp_name in pipeline.pretrained_component_names:
            if comp_name in components_update_dict:
                continue  # Already provided externally
                
            comp_spec = pipeline.get_component_spec(comp_name)
            comp_with_same_load_id = components._lookup_ids(load_id=comp_spec.load_id)
            if comp_with_same_load_id:
                # Reuse existing component
                comp_id = list(comp_with_same_load_id)[0]
                components_update_dict[comp_name] = components.get_one(component_id=comp_id)
            else:
                components_to_load.append(comp_name)

        pipeline.update_components(**components_update_dict)
        pipeline.load_components(names=components_to_load, torch_dtype=torch_dtype)
        
        # Move to device if not using auto offload
        if not auto_offload:
            pipeline.to(device)
        
        # Build inputs dict
        inputs_dict = {}
        for input_name in node_config["input_names"]:
            if input_name in kwargs:
                inputs_dict[input_name] = kwargs.pop(input_name)
        
        # Execute pipeline - strip out_ prefix for pipeline call
        mellon_output_names = node_config["output_names"]
        pipeline_output_names = [name[4:] if name.startswith("out_") else name for name in mellon_output_names]
        pipeline_outputs = pipeline(**inputs_dict, output=pipeline_output_names)
        
        # Map pipeline outputs back to Mellon names (with out_ prefix)
        final_outputs = {}
        for mellon_name, pipeline_name in zip(mellon_output_names, pipeline_output_names):
            if pipeline_name in pipeline_outputs:
                final_outputs[mellon_name] = pipeline_outputs[pipeline_name]
        
        final_outputs["doc"] = pipeline.blocks.doc
        return final_outputs
