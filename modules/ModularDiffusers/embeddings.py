import logging

from diffusers.modular_pipelines import ModularPipeline

from mellon.NodeBase import NodeBase

from . import components


logger = logging.getLogger("mellon")


class EncodePrompt(NodeBase):
    label = "Encode Prompt"
    category = "embedding"
    resizable = True
    params = {
        "text_encoders": {
            "label": "Text Encoders",
            "type": "diffusers_auto_models",
            "display": "input",
            "onSignal": {
                "QwenImageEditModularPipeline": ["image"],
                "QwenImageEditPlusModularPipeline": ["image"],
                "": [],
            },
        },
        "prompt": {"label": "Prompt", "type": "string", "default": "", "display": "textarea"},
        "image": {"label": "Image", "type": "image", "display": "input"},
        "negative_prompt": {"label": "Negative Prompt", "type": "string", "default": "", "display": "textarea"},
        "embeddings": {"label": "Text Embeddings", "display": "output", "type": "embeddings"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._text_encoder_node = None

    def execute(self, text_encoders, prompt, image, negative_prompt):
        logger.debug(f" EncodePrompt ({self.node_id}) received parameters:")
        logger.debug(f" - text_encoders: {text_encoders}")
        logger.debug(f" - image: {image}")
        logger.debug(f" - prompt: {prompt}")
        logger.debug(f" - negative_prompt: {negative_prompt}")

        text_encoders = text_encoders.copy()
        repo_id = text_encoders.pop("repo_id")
        text_blocks = ModularPipeline.from_pretrained(repo_id, components_manager=components).blocks.sub_blocks.pop(
            "text_encoder"
        )
        self._text_encoder_node = text_blocks.init_pipeline(repo_id, components_manager=components)

        text_encoder_components = {
            text_component_name: components.get_one(text_encoders[text_component_name]["model_id"])
            for text_component_name in text_encoders.keys()
        }

        self._text_encoder_node.update_components(**text_encoder_components)

        text_node_kwargs = {}

        if image is not None and "image" in text_blocks.input_names:
            text_node_kwargs["image"] = image

        text_node_kwargs.update(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }
        )

        text_state = self._text_encoder_node(**text_node_kwargs)
        # YiYi TODO: update in diffusers so that always use denoiser_input_fields
        text_embeddings = text_state.get_by_kwargs("denoiser_input_fields")

        return {"embeddings": text_embeddings}
