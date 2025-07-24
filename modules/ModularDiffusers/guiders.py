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
            guider_options[key] = value

        logger.debug(f" - guider options: {guider_options}")

        guider_cls = getattr(__import__("diffusers", fromlist=[guider]), guider)
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
        "block_name": {
            "label": "Fully Qualified Name",
            "type": "string",
            "value": "mid_block.attentions.0.transformer_blocks",
            "hidden": True,
        },
        "dropout": {
            "label": "Mid Block",
            "type": "float",
            "display": "slider",
            "value": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "indices": {"label": "Indices", "type": "string", "value": "2,9", "display": "textarea"},
        "skip_attention": {
            "label": "Skip Attention",
            "type": "boolean",
            "value": True,
        },
        "skip_attention_scores": {
            "label": "Skip Attention Scores",
            "type": "boolean",
            "value": False,
        },
        "skip_ff": {
            "label": "Skip feed-forward blocks",
            "type": "boolean",
            "value": False,
        },
        "layers_config": {"label": "Layers", "display": "output", "type": "layers_config"},
    }

    def execute(self, **kwargs):
        # Parse indices from string to list of integers
        indices_str = kwargs.get("indices", "2,9")
        indices = [int(x.strip()) for x in indices_str.split(",")]

        # Get fqn from kwargs
        fqn = kwargs.get("fqn", "mid_block.attentions.0.transformer_blocks")

        # Get optional boolean parameters with defaults
        skip_attention = kwargs.get("skip_attention", False)
        skip_attention_scores = kwargs.get("skip_attention_scores", True)
        skip_ff = kwargs.get("skip_ff", False)

        # Build LayerSkipConfig with parsed values
        layer_config = LayerSkipConfig(
            indices=indices,
            fqn=fqn,
            skip_attention=skip_attention,
            skip_attention_scores=skip_attention_scores,
            skip_ff=skip_ff,
        )

        return {"layers_config": layer_config}
