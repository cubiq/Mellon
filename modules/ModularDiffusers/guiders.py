import logging

from diffusers import LayerSkipConfig

from mellon.NodeBase import NodeBase


logger = logging.getLogger("mellon")

GUIDER_CONFIGS = {
    "PerturbedAttentionGuidance": {
        "perturbed_guidance_scale": {
            "label": "Perturbed Guidance Scale",
            "type": "float",
            "value": 2.8,
            "min": 0.0,
            "max": 10.0,
        },
        "perturbed_guidance_start": {
            "label": "Perturbed Guidance Start",
            "type": "float",
            "display": "slider",
            "default": 0.01,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "perturbed_guidance_stop": {
            "label": "Stop",
            "type": "float",
            "display": "slider",
            "default": 0.2,
            "min": 0.02,
            "max": 1.0,
            "step": 0.01,
        },
        "layers_config": {"label": "Layers", "type": "layers_config", "display": "input"},
    }
}


class Guider(NodeBase):
    label = "Guider"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    params = {
        "guider": {
            "label": "Guider",
            "fieldOptions": {"loading": True},
            "type": "string",
            "options": {
                "ClassifierFreeGuidance": "Classifier Free Guidance",
                "PerturbedAttentionGuidance": "Perturbed Attention Guidance",
            },
            "value": "ClassifierFreeGuidance",
            "onChange": "updateNode",
        },
        "guidance_scale": {
            "label": "Guidance Scale",
            "type": "float",
            "display": "slider",
            "value": 5,
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
        },
        "guidance_rescale": {
            "label": "Guidance Rescale",
            "type": "float",
            "display": "slider",
            "value": 0.0,
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
        },
        "use_original_formulation": {
            "label": "Original Formulation",
            "type": "boolean",
            "value": False,
        },
        "start": {
            "label": "Start",
            "type": "float",
            "display": "slider",
            "value": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "stop": {
            "label": "Stop",
            "type": "float",
            "display": "slider",
            "value": 1.0,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01,
        },
        "guider_out": {"label": "Guider", "display": "output", "type": "guider"},
    }

    def updateNode(self, values, ref):
        value = values.get("guider")

        params = GUIDER_CONFIGS.get(value, {})
        self.send_node_definition(params)

    def execute(self, guider, layers_config=None, **kwargs):
        logger.debug(f" Guider ({self.node_id}) received parameters:")
        logger.debug(f" - guider: {guider}")
        logger.debug(f" - kwargs: {kwargs}")

        guider_options = {}

        for key, value in kwargs.items():
            if key == "use_original_formulation":
                guider_options[key] = value
            else:
                guider_options[key] = float(value)

        logger.debug(f" - guider options: {guider_options}")

        guider_cls = getattr(__import__("diffusers", fromlist=[guider]), guider)

        if layers_config is None:
            # create defaults
            # TODO: need a way to find model arch for blocks and layers
            configs = {
                "PerturbedAttentionGuidance": {
                    "perturbed_guidance_config": LayerSkipConfig(
                        indices=[0],
                        fqn="mid_block.attentions.0.transformer_blocks",
                        dropout=1.0,
                        skip_attention=False,
                        skip_attention_scores=True,
                        skip_ff=False,
                    )
                }
            }
        else:
            configs = {"PerturbedAttentionGuidance": {"perturbed_guidance_config": layers_config}}

        options = {**guider_options}

        if guider in configs:
            options.update(configs[guider])

        guider_out = guider_cls(**options)

        return {"guider_out": guider_out}


class Layers(NodeBase):
    label = "Layers"
    category = "sampler"
    resizable = True
    params = {
        "select_multi": {
            "label": "Blocks",
            "type": "string",
            "display": "select",
            "options": {
                "down_blocks.1.attentions.0": "DownBlocks.1.Attentions.0",
                "down_blocks.1.attentions.1": "DownBlocks.1.Attentions.1",
                "down_blocks.2.attentions.0": "DownBlocks.2.Attentions.0",
                "down_blocks.2.attentions.1": "DownBlocks.2.Attentions.1",
                "mid_block.attentions.0": "MidBlock.Attentions.0",
                "up_blocks.0.attentions.0": "UpBlocks.0.Attentions.0",
                "up_blocks.0.attentions.1": "UpBlocks.0.Attentions.1",
                "up_blocks.0.attentions.2": "UpBlocks.0.Attentions.2",
                "up_blocks.1.attentions.0": "UpBlocks.1.Attentions.0",
                "up_blocks.1.attentions.1": "UpBlocks.1.Attentions.1",
                "up_blocks.1.attentions.2": "UpBlocks.1.Attentions.2",
            },
            "default": ["mid_block.attentions.0"],
            "fieldOptions": {"multiple": True},
            "onChange": {
                "down_blocks.1.attentions.0": ["down_blocks.1.attentions.0"],
                "down_blocks.1.attentions.1": ["down_blocks.1.attentions.1"],
                "down_blocks.2.attentions.0": ["down_blocks.2.attentions.0"],
                "down_blocks.2.attentions.1": ["down_blocks.2.attentions.1"],
                "mid_block.attentions.0": ["mid_block.attentions.0"],
                "up_blocks.0.attentions.0": ["up_blocks.0.attentions.0"],
                "up_blocks.0.attentions.1": ["up_blocks.0.attentions.1"],
                "up_blocks.0.attentions.2": ["up_blocks.0.attentions.2"],
                "up_blocks.1.attentions.0": ["up_blocks.1.attentions.0"],
                "up_blocks.1.attentions.1": ["up_blocks.1.attentions.1"],
                "up_blocks.1.attentions.2": ["up_blocks.1.attentions.2"],
            },
        },
        "down_blocks.1.attentions.0": {"label": "DownBlocks.1.0 Layers", "type": "string", "default": "0"},
        "down_blocks.1.attentions.1": {"label": "DownBlocks.1.1 Layers", "type": "string", "default": "0"},
        "down_blocks.2.attentions.0": {"label": "DownBlocks.2.0 Layers", "type": "string", "default": "0"},
        "down_blocks.2.attentions.1": {"label": "DownBlocks.2.1 Layers", "type": "string", "default": "0"},
        "mid_block.attentions.0": {"label": "MidBlock.0 Layers", "type": "string", "default": "0"},
        "up_blocks.0.attentions.0": {"label": "UpBlocks.0.0 Layers", "type": "string", "default": "0"},
        "up_blocks.0.attentions.1": {"label": "UpBlocks.0.1 Layers", "type": "string", "default": "0"},
        "up_blocks.0.attentions.2": {"label": "UpBlocks.0.2 Layers", "type": "string", "default": "0"},
        "up_blocks.1.attentions.0": {"label": "UpBlocks.1.0 Layers", "type": "string", "default": "0"},
        "up_blocks.1.attentions.1": {"label": "UpBlocks.1.1 Layers", "type": "string", "default": "0"},
        "up_blocks.1.attentions.2": {"label": "UpBlocks.1.2 Layers", "type": "string", "default": "0"},
        "layers_config": {"label": "Layers", "display": "output", "type": "layers_config"},
    }

    def execute(self, **kwargs):
        layer_configs = []

        selected_blocks = kwargs.get("select_multi", [])

        for block_name in selected_blocks:
            indices_str = kwargs.get(block_name, "0")
            indices = [int(x.strip()) for x in indices_str.split(",")]
            layer_configs.append(
                LayerSkipConfig(
                    indices=indices,
                    fqn=f"{block_name}.transformer_blocks",
                    dropout=1.0,
                    skip_attention=False,
                    skip_attention_scores=True,
                    skip_ff=False,
                )
            )

        print(f"{layer_configs=}")

        return {"layers_config": layer_configs}


# class LayersFull(NodeBase):
#     label = "LayersFull"
#     category = "sampler"
#     resizable = True
#     params = {
#         "down_blocks.1.attentions.0.transformer_blocks": {
#             "label": "DownBlocks.1.Attentions.0",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": True, "skip_checkboxes_visible": True},
#         },
#         "down_blocks.1.attentions.1.transformer_blocks": {
#             "label": "DownBlocks.1.Attentions.1",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": True, "skip_checkboxes_visible": True},
#         },
#         "down_blocks.2.attentions.0.transformer_blocks": {
#             "label": "DownBlocks.2.Attentions.0",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": False, "skip_checkboxes_visible": True},
#         },
#         "down_blocks.2.attentions.1.transformer_blocks": {
#             "label": "DownBlocks.2.Attentions.1",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": True, "skip_checkboxes_visible": True},
#         },
#         "mid_block.attentions.0.transformer_blocks": {
#             "label": "MidBlock.Attentions.0",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": False, "skip_checkboxes_visible": False},
#         },
#         "up_blocks.0.attentions.0.transformer_blocks": {
#             "label": "UpBlocks.0.attentions.0",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": False, "skip_checkboxes_visible": False},
#         },
#         "up_blocks.0.attentions.1.transformer_blocks": {
#             "label": "UpBlocks.0.attentions.1",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": False, "skip_checkboxes_visible": False},
#         },
#         "up_blocks.0.attentions.2.transformer_blocks": {
#             "label": "UpBlocks.0.attentions.2",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": False, "skip_checkboxes_visible": False},
#         },
#         "up_blocks.1.attentions.0.transformer_blocks": {
#             "label": "UpBlocks.1.attentions.0",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": False, "skip_checkboxes_visible": False},
#         },
#         "up_blocks.1.attentions.1.transformer_blocks": {
#             "label": "UpBlocks.1.attentions.1",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": False, "skip_checkboxes_visible": False},
#         },
#         "up_blocks.1.attentions.2.transformer_blocks": {
#             "label": "UpBlocks.1.attentions.2",
#             "display": "custom.LayerField",
#             "value": {"enabled": False, "indices": "", "dropout_visible": False, "skip_checkboxes_visible": False},
#         },
#         "layers_config": {"label": "Layers", "display": "output", "type": "layers_config"},
#     }

#     def execute(self, **kwargs):
#         layer_configs = []

#         for fqn, config in kwargs.items():
#             if not config.get("enabled", False):
#                 continue

#             indices_str = config.get("indices", "")
#             indices = []
#             if indices_str.strip():
#                 indices = [int(x.strip()) for x in indices_str.split(",")]
#             else:
#                 indices = [0]

#             layer_config = LayerSkipConfig(
#                 indices=indices,
#                 fqn=fqn,
#                 dropout=config.get("dropout", 1.0),
#                 skip_attention=config.get("skip_attention", False),
#                 skip_attention_scores=config.get("skip_attention_scores", True),
#                 skip_ff=config.get("skip_ff", False),
#             )

#             layer_configs.append(layer_config)

#         return {"layers_config": layer_configs}
