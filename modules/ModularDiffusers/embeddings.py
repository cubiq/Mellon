import logging

from diffusers.modular_pipelines import ModularPipeline

from mellon.NodeBase import NodeBase
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST

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
            "onSignal": "on_signal",
        },
        "prompt": {"label": "Prompt", "type": "string", "default": "", "display": "textarea"},
        "image": {"label": "Image", "type": "image", "display": "input", "visible": False},
        "negative_prompt": {"label": "Negative Prompt", "type": "string", "default": "", "display": "textarea"},
        "embeddings": {"label": "Text Embeddings", "display": "output", "type": "embeddings"},
        "device": {
            "label": "Device",
            "type": "string",
            "value": DEFAULT_DEVICE,
            "options": DEVICE_LIST,
            "visible": False,
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._text_encoder_node = None
        self._auto_cpu_offload = True
        self._image_visible = False

    def on_signal(self, values, ref):
        try:
            signal_value = self.get_signal_value("text_encoders")

            image_visible = signal_value in {"QwenImageEditModularPipeline", "QwenImageEditPlusModularPipeline"}
            auto_cpu_offload = True

            if signal_value is None:
                image_visible = False
            elif isinstance(signal_value, bool):
                auto_cpu_offload = signal_value
            elif isinstance(signal_value, str):
                if signal_value.lower() == "true":
                    auto_cpu_offload = True
                elif signal_value.lower() == "false":
                    auto_cpu_offload = False

            self._image_visible = image_visible
            self._auto_cpu_offload = auto_cpu_offload

            self.set_field_visibility({"image": image_visible})
            self.set_field_visibility({"device": not auto_cpu_offload})
        except ValueError:
            pass

    def execute(self, text_encoders, prompt, image, negative_prompt, device):
        logger.debug(f" EncodePrompt ({self.node_id}) received parameters:")
        logger.debug(f" - text_encoders: {text_encoders}")
        logger.debug(f" - image: {image}")
        logger.debug(f" - prompt: {prompt}")
        logger.debug(f" - negative_prompt: {negative_prompt}")
        logger.debug(f" - device: {device}")

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

        if not self._auto_cpu_offload:
            self._text_encoder_node.to(device)

        text_state = self._text_encoder_node(**text_node_kwargs)
        text_embeddings = text_state.get_by_kwargs("denoiser_input_fields")

        return {"embeddings": text_embeddings}
