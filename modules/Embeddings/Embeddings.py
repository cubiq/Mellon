from mellon.NodeBase import NodeBase
import torch

class CombineEmbeddings(NodeBase):
    def execute(self, embeddings_1, embeddings_2, ratio):
        out = {}
        if 'prompt_embeds' in embeddings_1 and 'prompt_embeds' in embeddings_2:
            embeds_1 = embeddings_1['prompt_embeds']
            embeds_2 = embeddings_2['prompt_embeds']
            out['prompt_embeds'] = embeds_1 * ratio + embeds_2 * (1 - ratio)
        
        if 'pooled_prompt_embeds' in embeddings_1 and 'pooled_prompt_embeds' in embeddings_2:
            pooled_embeds_1 = embeddings_1['pooled_prompt_embeds']
            pooled_embeds_2 = embeddings_2['pooled_prompt_embeds']
            out['pooled_prompt_embeds'] = pooled_embeds_1 * ratio + pooled_embeds_2 * (1 - ratio)

        if 'negative_prompt_embeds' in embeddings_1 and 'negative_prompt_embeds' in embeddings_2:
            negative_embeds_1 = embeddings_1['negative_prompt_embeds']
            negative_embeds_2 = embeddings_2['negative_prompt_embeds']
            out['negative_prompt_embeds'] = negative_embeds_1 * ratio + negative_embeds_2 * (1 - ratio)

        if 'negative_pooled_prompt_embeds' in embeddings_1 and 'negative_pooled_prompt_embeds' in embeddings_2:
            negative_pooled_embeds_1 = embeddings_1['negative_pooled_prompt_embeds']
            negative_pooled_embeds_2 = embeddings_2['negative_pooled_prompt_embeds']
            out['negative_pooled_prompt_embeds'] = negative_pooled_embeds_1 * ratio + negative_pooled_embeds_2 * (1 - ratio)

        return { 'embeddings_out': out }
