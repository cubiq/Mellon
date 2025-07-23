import logging

from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS

from mellon.NodeBase import NodeBase

from . import components


logger = logging.getLogger("mellon")

t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(ALL_BLOCKS["text2img"])
text_blocks = t2i_blocks.sub_blocks.pop("text_encoder")


class EncodePrompt(NodeBase):
    label = "Encode Prompt"
    category = "embedding"
    resizable = True
    params = {
        "text_encoders": {"label": "Text Encoders", "type": "text_encoders", "display": "input"},
        "prompt": {"label": "Prompt", "type": "string", "default": "", "display": "textarea"},
        "negative_prompt": {"label": "Negative Prompt", "type": "string", "default": "", "display": "textarea"},
        "embeddings": {"label": "Text Embeddings", "display": "output", "type": "embeddings"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._text_encoder_node = text_blocks.init_pipeline(components_manager=components)

    def execute(self, text_encoders, prompt, negative_prompt):
        logger.debug(f" EncodePrompt ({self.node_id}) received parameters:")
        logger.debug(f" - text_encoders: {text_encoders}")

        text_encoder_components = {
            "text_encoder": components.get_one(text_encoders["text_encoder"]["model_id"]),
            "text_encoder_2": components.get_one(text_encoders["text_encoder_2"]["model_id"]),
            "tokenizer": components.get_one(text_encoders["tokenizer"]["model_id"]),
            "tokenizer_2": components.get_one(text_encoders["tokenizer_2"]["model_id"]),
        }

        self._text_encoder_node.update_components(**text_encoder_components)

        text_embeddings = self._text_encoder_node(
            prompt=prompt,
            negative_prompt=negative_prompt,
            output=[
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            ],
        )

        return {"embeddings": text_embeddings}
