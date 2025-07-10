import logging

import torch
from diffusers import ComponentsManager, ComponentSpec, ModularPipeline
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS

from mellon.NodeBase import NodeBase, deep_equal
from utils.torch_utils import str_to_dtype

logger = logging.getLogger("mellon")
logger.setLevel(logging.DEBUG)


def node_get_component_id(node_id=None, manager=None, name=None):
    comp_ids = manager._lookup_ids(name=name, collection=node_id)
    if len(comp_ids) != 1:
        raise ValueError(
            f"Expected 1 component for {name} for node {node_id}, got {len(comp_ids)}"
        )
    return list(comp_ids)[0]


def node_get_component_info(node_id=None, manager=None, name=None):
    comp_id = node_get_component_id(node_id, manager, name)
    return manager.get_model_info(comp_id)


def has_changed(old_params, new_params):
    for key in new_params:
        new_value = new_params.get(key)
        old_value = old_params.get(key)
        if new_value is not None and key not in old_params:
            return True
        if not deep_equal(old_value, new_value):
            return True
    return False


t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(ALL_BLOCKS["text2img"])
text_blocks = t2i_blocks.sub_blocks.pop("text_encoder")
decoder_blocks = t2i_blocks.sub_blocks.pop("decode")

components = ComponentsManager()

# TODO: make this a user selectable option or automatic depending on VRAM
components.enable_auto_cpu_offload(device="cuda")


class AutoModelLoader(NodeBase):
    label = "Load Model"
    category = "Modular Diffusers"
    resizable = True
    params = {
        "name": {
            "label": "Name",
            "type": "string",
        },
        "model_id": {
            "label": "Model ID",
            "type": "string",
        },
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float16",
            "postProcess": str_to_dtype,
        },
        "subfolder": {
            "label": "Subfolder",
            "type": "string",
            "default": "",
        },
        "variant": {
            "type": "string",
            "default": "",
            "options": ["", "fp16", "bf16"],
        },
        "model": {
            "label": "Model",
            "display": "output",
            "type": "diffusers_auto_model",
        },
    }

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove(comp_id)
        super().__del__()

    def execute(self, name, model_id, dtype, variant=None, subfolder=None):
        logger.debug(f"AutoModelLoader ({self.node_id}) received parameters:")
        logger.debug(f"  name: '{name}'")
        logger.debug(f"  model_id: '{model_id}'")
        logger.debug(f"  subfolder: '{subfolder}'")
        logger.debug(f"  variant: '{variant}'")
        logger.debug(f"  dtype: '{dtype}'")

        # Normalize parameters
        variant = None if variant == "" else variant
        subfolder = None if subfolder == "" else subfolder

        spec = ComponentSpec(
            name=name, repo=model_id, subfolder=subfolder, variant=variant
        )
        model = spec.load(torch_dtype=dtype)
        comp_id = components.add(name, model, collection=self.node_id)
        logger.debug(f" AutoModelLoader: comp_id added: {comp_id}")
        logger.debug(f" AutoModelLoader: component manager: {components}")

        return {"model": components.get_model_info(comp_id)}


class ModelsLoader(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.loader = None

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove(comp_id)
        self.loader = None
        super().__del__()

    def __call__(self, **kwargs):
        self._old_params = self.params.copy()
        return super().__call__(**kwargs)

    def execute(self, repo_id, device, dtype, unet=None, vae=None):
        logger.debug(f"""
            ModelsLoader ({self.node_id}) received parameters:
            old_params: {self._old_params}
            new params:
            - repo_id: {repo_id}
            - dtype: {dtype}
            - device: {device}
        """)

        repo_changed = has_changed(
            self._old_params,
            {
                "repo_id": repo_id,
                "dtype": dtype,
            },
        )
        unet_input_changed = has_changed(self._old_params, {"unet": unet})
        vae_input_changed = has_changed(self._old_params, {"vae": vae})

        logger.debug(
            f"Changes detected - repo: {repo_changed}, unet: {unet_input_changed}, vae: {vae_input_changed}, "
        )

        self.loader = ModularPipeline.from_pretrained(
            repo_id, components_manager=components, collection=self.node_id
        )

        if repo_changed:
            self.loader.load_components(
                names=[
                    "scheduler",
                    "text_encoder",
                    "text_encoder_2",
                    "tokenizer",
                    "tokenizer_2",
                ],
                torch_dtype=dtype,
            )

        loaded_components = {}

        # see if we can just use a growing input that we can just attach models
        # and then load them we don't need `unet` or `vae` here, they will
        # get automatically loaded depending on the type of model, even
        # ip adapters or controlnets, this is the first step to make it a
        # one in all wonder node
        if unet is None:
            if repo_changed or unet_input_changed:
                self.loader.load_components("unet", torch_dtype=dtype)
        else:
            unet_component = components.get_one(unet["model_id"])
            self.loader.update_components(unet=unet_component)

        if vae is None:
            if repo_changed or vae_input_changed:
                self.loader.load_components("vae", torch_dtype=dtype)
        else:
            new_vae = vae["model_id"]
            self.loader.update_components(**new_vae)

        if repo_changed:
            components.enable_auto_cpu_offload(device=device)

        # Construct loaded_components at the end after all modifications
        loaded_components = {
            "unet_out": node_get_component_info(
                node_id=self.node_id, manager=components, name="unet"
            ),
            "vae_out": node_get_component_info(
                node_id=self.node_id, manager=components, name="vae"
            ),
            "text_encoders": {
                k: node_get_component_info(
                    node_id=self.node_id, manager=components, name=k
                )
                for k in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
            },
            "scheduler": node_get_component_info(
                node_id=self.node_id, manager=components, name="scheduler"
            ),
        }

        logger.debug(f" ModelsLoader: Final component_manager state: {components}")

        return loaded_components


class EncodePrompt(NodeBase):
    label = "Encode Prompt"
    category = "Modular Diffusers"
    resizable = True
    params = {
        "text_encoders": {
            "label": "Text Encoders",
            "type": "text_encoders",
            "display": "input",
        },
        "prompt": {
            "label": "Prompt",
            "type": "string",
            "default": "",
            "display": "textarea",
        },
        "negative_prompt": {
            "label": "Negative Prompt",
            "type": "string",
            "default": "",
            "display": "textarea",
        },
        "embeddings": {
            "label": "Text Embeddings",
            "display": "output",
            "type": "embeddings",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._text_encoder_node = text_blocks.init_pipeline(
            components_manager=components
        )

    def execute(self, text_encoders, prompt, negative_prompt):
        logger.debug(f" EncodePrompt ({self.node_id}) received parameters:")
        logger.debug(f" - text_encoders: {text_encoders}")

        text_encoder_components = {
            "text_encoder": components.get_one(
                text_encoders["text_encoder"]["model_id"]
            ),
            "text_encoder_2": components.get_one(
                text_encoders["text_encoder_2"]["model_id"]
            ),
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


class Denoise(NodeBase):
    label = "Denoise"
    category = "Modular Diffusers"
    resizable = True
    params = {
        "unet": {
            "label": "Unet",
            "display": "input",
            "type": "diffusers_auto_model",
        },
        "scheduler": {
            "label": "Scheduler",
            "display": "input",
            "type": "scheduler",
        },
        "embeddings": {
            "label": "Text Embeddings",
            "display": "input",
            "type": "embeddings",
        },
        "width": {
            "label": "Width",
            "type": "int",
            "default": 1024,
            "min": 64,
            "step": 8,
        },
        "height": {
            "label": "Height",
            "type": "int",
            "default": 1024,
            "min": 64,
            "step": 8,
        },
        "seed": {
            "label": "Seed",
            "type": "int",
            "display": "random",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "steps": {
            "label": "Steps",
            "type": "int",
            "display": "slider",
            "default": 30,
            "min": 1,
            "max": 100,
        },
        "cfg": {
            "label": "Guidance",
            "type": "float",
            "display": "slider",
            "default": 5,
            "min": 0.0,
            "max": 50.0,
            "step": 0.5,
        },
        "latents": {
            "label": "Latents",
            "type": "latents",
            "display": "output",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._denoise_node = t2i_blocks.init_pipeline(components_manager=components)

    def execute(self, unet, scheduler, width, height, embeddings, seed, steps, cfg):
        logger.debug(f" Denoise ({self.node_id}) received parameters:")
        logger.debug(f" - unet: {unet}")
        logger.debug(f" - scheduler: {scheduler}")
        logger.debug(f" - embeddings: {embeddings}")
        logger.debug(f" - width: {width}")
        logger.debug(f" - height: {height}")
        logger.debug(f" - seed: {seed}")
        logger.debug(f" - steps: {steps}")
        logger.debug(f" - cfg: {cfg}")

        unet_component = components.get_one(unet["model_id"])
        scheduler_component = components.get_one(scheduler["model_id"])
        self._denoise_node.update_components(
            unet=unet_component, scheduler=scheduler_component
        )

        generator = torch.Generator(device="cuda").manual_seed(seed)

        guider_spec = self._denoise_node.get_component_spec("guider")
        guider_spec.config["guidance_scale"] = cfg
        self._denoise_node.update_components(guider=guider_spec)

        latents = self._denoise_node(
            **embeddings,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
            output="latents",
        )

        return {"latents": latents}


class DecodeLatents(NodeBase):
    label = "Decode Latents"
    category = "Modular Diffusers"
    resizable = True
    params = {
        "vae": {
            "label": "VAE",
            "display": "input",
            "type": "diffusers_auto_model",
        },
        "latents": {
            "label": "Latents",
            "type": "latents",
            "display": "input",
        },
        "images": {
            "label": "Images",
            "type": "image",
            "display": "output",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._decoder_node = decoder_blocks.init_pipeline(components_manager=components)

    def execute(self, vae, latents):
        logger.debug(f" DecodeLatents ({self.node_id}) received parameters:")
        logger.debug(f" - vae: {vae}")
        logger.debug(f" - latents: {latents.shape}")

        vae_component = components.get_one(vae["model_id"])
        self._decoder_node.update_components(vae=vae_component)
        image = self._decoder_node(latents=latents, output="images")[0]

        return {"images": image}
