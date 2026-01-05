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
        custom_params = custom_mellon_config["params"]

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
        mellon_dict = MellonNodeConfig.load_mellon_config(repo_id)
        mellon_mixin = MellonNodeConfig.from_mellon_dict(mellon_dict)

        # update the components for the controlnet node
        components_update_dict = {}
        components_load_list = []
        for comp_name in pipeline.pretrained_component_names:
            if comp_name in kwargs:
                model_info_dict = kwargs.pop(comp_name)
                components_update_dict.update(components.get_components_by_ids([model_info_dict["model_id"]]))
            else:
                components_load_list.append(comp_name)

        pipeline.load_components(names=components_load_list)
        pipeline.update_components(**components_update_dict)

        inputs_dict = {}
        for input_name in pipeline.blocks.input_names:
            if input_name in kwargs and input_name not in inputs_dict:
                inputs_dict[input_name] = kwargs.pop(input_name)

        output_names = [output_name for output_name in mellon_mixin.outputs.keys()]
        pipeline_outputs = pipeline(**inputs_dict, output=output_names)

        for k, v in kwargs.items():
            if k in output_names and k not in pipeline_outputs:
                pipeline_outputs[k] = v
        pipeline_outputs["doc"] = pipeline.blocks.doc
        return pipeline_outputs
