import logging

from diffusers import LayerSkipConfig, SmoothedEnergyGuidanceConfig

from mellon.NodeBase import NodeBase

from . import FLUX_BLOCKS, QWEN_IMAGE_BLOCKS, SDXL_BLOCKS


logger = logging.getLogger("mellon")

LAYER_CONFIG_MAPPING = {
    "SkipLayerGuidance": "skip_layer_config",
    "AutoGuidance": "auto_guidance_config",
    "SmoothedEnergyGuidance": "seg_guidance_config",
}

# TODO: not sure if these defaults make sense and it would be more complex with each model
DEFAULT_CONFIGS = {
    "SkipLayerGuidance": {
        "skip_layer_config": LayerSkipConfig(
            indices=[0],
            fqn="mid_block.attentions.0.transformer_blocks",
            dropout=1.0,
            skip_attention=False,
            skip_attention_scores=True,
            skip_ff=False,
        )
    },
    "AutoGuidance": {
        "auto_guidance_config": LayerSkipConfig(
            indices=[0],
            fqn="mid_block.attentions.0.transformer_blocks",
            dropout=1.0,
            skip_attention=False,
            skip_attention_scores=True,
            skip_ff=False,
        )
    },
    "SmoothedEnergyGuidance": {
        "seg_guidance_config": SmoothedEnergyGuidanceConfig(
            indices=[0],
            fqn="mid_block.attentions.0.transformer_blocks",
        )
    },
}

GUIDER_CONFIGS = {
    "SkipLayerGuidance": {
        "skip_layer_guidance_scale": {
            "label": "Skip Layer Guidance Scale",
            "type": "float",
            "value": 2.8,
            "min": 0.0,
            "max": 10.0,
        },
        "skip_layer_guidance_start": {
            "label": "Skip Layer Start",
            "type": "float",
            "display": "slider",
            "default": 0.01,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "skip_layer_guidance_stop": {
            "label": "Stop",
            "type": "float",
            "display": "slider",
            "default": 0.2,
            "min": 0.02,
            "max": 1.0,
            "step": 0.01,
        },
    },
    "AdaptiveProjectedGuidance": {
        "adaptive_projected_guidance_momentum": {
            "label": "Adaptive Projected Guidance Momentum",
            "type": "float",
            "value": 0.0,
            "min": -1.0,
            "max": 1.0,
            "step": 0.01,
        },
        "adaptive_projected_guidance_rescale": {
            "label": "Adaptive Projected Guidance Rescale",
            "type": "float",
            "value": 15.0,
            "min": 0.0,
            "max": 100.0,
        },
    },
    "ClassifierFreeZeroStarGuidance": {
        "zero_init_steps": {
            "label": "Zero Init Steps",
            "type": "int",
            "value": 1,
        },
    },
    "AutoGuidance": {
        "dropout": {
            "label": "Dropout",
            "type": "float",
            "value": 1.0,
            "min": 0.0,
            "max": 10.0,
        }
    },
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
                "SkipLayerGuidance": "Skip Layer Guidance",
                "AdaptiveProjectedGuidance": "Adaptive Projected Guidance",
                "ClassifierFreeZeroStarGuidance": "Classifier Free Zero Star Guidance",
                "AutoGuidance": "Auto Guidance",
                "SmoothedEnergyGuidance": "Smoothed Energy Guidance",
                "TangentialClassifierFreeGuidance": "Tangential Classifier Free Guidance",
                "FrequencyDecoupledGuidance": "Frequency Decoupled Guidance",
            },
            "value": "ClassifierFreeGuidance",
            "onChange": [
                "updateNode",
                {
                    "SkipLayerGuidance": ["layers_config"],
                    "AutoGuidance": ["layers_config"],
                    "SmoothedEnergyGuidance": ["layers_config"],
                },
            ],
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
        "guider_out": {
            "label": "Guider",
            "display": "output",
            "type": "custom_guider",
            "onSignal": {
                "action": "signal",
                "target": "layers_config",
            },
        },
        "layers_config": {"label": "Layers", "type": "layers_config", "display": "input"},
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

        configs = {}

        if guider in LAYER_CONFIG_MAPPING:
            if layers_config is None:
                configs[guider] = DEFAULT_CONFIGS.get(guider, {})
            else:
                config_arg_name = LAYER_CONFIG_MAPPING[guider]

                if isinstance(layers_config, list):
                    layer_configs = []

                    for config_dict in layers_config:
                        if guider == "SmoothedEnergyGuidance":
                            layer_config = SmoothedEnergyGuidanceConfig(
                                indices=config_dict["indices"], fqn=config_dict["fqn"]
                            )
                        else:
                            layer_config = LayerSkipConfig(**config_dict)

                        layer_configs.append(layer_config)

                    configs[guider] = {config_arg_name: layer_configs}
                else:
                    configs[guider] = {config_arg_name: layers_config}

        options = {**guider_options}

        if guider in configs:
            options.update(configs[guider])

        guider_out = guider_cls(**options)

        return {"guider_out": guider_out}


class Layers(NodeBase):
    label = "Layers"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    params = {
        "blocks_select": {
            "label": "Blocks",
            "type": "string",
            "display": "select",
            "options": {"": ""},
            "fieldOptions": {"multiple": True},
            "onChange": "set_blocks",
        },
        "layers_config": {
            "label": "Layers",
            "display": "output",
            "type": "layers_config",
            "onSignal": {
                "action": "value",
                "target": "blocks_select",
                "prop": "options",
                "data": {
                    "StableDiffusionXLModularPipeline": SDXL_BLOCKS,
                    "QwenImageModularPipeline": QWEN_IMAGE_BLOCKS,
                    "QwenImageEditModularPipeline": QWEN_IMAGE_BLOCKS,
                    "QwenImageEditPlusModularPipeline": QWEN_IMAGE_BLOCKS,
                    "FluxModularPipeline": FLUX_BLOCKS,
                },
            },
        },
    }

    def set_blocks(self, values, ref):
        blocks_select = values.get("blocks_select", [])

        params = {}

        for block_name in blocks_select:
            params[block_name] = {
                "label": block_name,
                "display": "layerconfig",
                "value": {"enabled": True, "indices": "", "dropout_visible": True, "skip_checkboxes_visible": True},
            }
        self.send_node_definition(params)

    def execute(self, **kwargs):
        layer_configs = []

        for block in kwargs:
            if block == "blocks_select":
                continue

            config = kwargs.get(block, {})

            indices_str = config.get("indices", "")
            indices = []
            if indices_str.strip():
                indices = [int(x.strip()) for x in indices_str.split(",")]
            else:
                indices = [0]

            layer_config = {
                "indices": indices,
                "fqn": f"{block}.transformer_blocks",
                "dropout": config.get("dropout", 1.0),
                "skip_attention": config.get("skip_attention", False),
                "skip_attention_scores": config.get("skip_attention_scores", False),
                "skip_ff": config.get("skip_ff", False),
            }

            layer_configs.append(layer_config)

        return {"layers_config": layer_configs}
