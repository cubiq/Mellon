import logging

from diffusers import ModularPipeline
from diffusers.modular_pipelines.mellon_node_utils import MellonPipelineConfig

from mellon.NodeBase import NodeBase

from . import components


logger = logging.getLogger("mellon")


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
        "doc": {
            "label": "Doc",
            "type": "string",
            "display": "output",
        },
    }

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
        pipeline = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True, components_manager=components)
        
        # Load config to get input/output names
        custom_mellon_config = MellonPipelineConfig.load(repo_id)
        node_config = custom_mellon_config.node_params["custom"]
        
        # Handle components
        components_update_dict = {}
        for comp_name in node_config.get("model_input_names", []):
            if comp_name in kwargs:
                model_info_dict = kwargs.pop(comp_name)
                components_update_dict.update(components.get_components_by_ids([model_info_dict["model_id"]]))
        pipeline.update_components(**components_update_dict)
        pipeline.load_components()
        
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
