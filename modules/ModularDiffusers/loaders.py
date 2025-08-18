import logging

import torch
from diffusers import ComponentSpec, ModularPipeline

from mellon.NodeBase import NodeBase
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

from . import components


logger = logging.getLogger("mellon")


def node_get_component_info(node_id=None, manager=None, name=None):
    comp_ids = manager._lookup_ids(name=name, collection=node_id)
    if len(comp_ids) != 1:
        raise ValueError(f"Expected 1 component for {name} for node {node_id}, got {len(comp_ids)}")
    return manager.get_model_info(list(comp_ids)[0])


# TODO: make this a user selectable option or automatic depending on VRAM
components.enable_auto_cpu_offload(device="cuda")


class AutoModelLoader(NodeBase):
    label = "Load Model"
    category = "loader"
    resizable = True
    params = {
        "name": {"label": "Name", "type": "string"},
        "model_id": {"label": "Model ID", "type": "string"},
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float16",
            "postProcess": str_to_dtype,
        },
        "subfolder": {"label": "Subfolder", "type": "string", "default": ""},
        "variant": {"type": "string", "default": "", "options": ["", "fp16", "bf16"]},
        "model": {"label": "Model", "display": "output", "type": "diffusers_auto_model"},
    }

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
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

        spec = ComponentSpec(name=name, repo=model_id, subfolder=subfolder, variant=variant)
        model = spec.load(torch_dtype=dtype)
        comp_id = components.add(name, model, collection=self.node_id)
        logger.debug(f" AutoModelLoader: comp_id added: {comp_id}")
        logger.debug(f" AutoModelLoader: component manager: {components}")

        return {"model": components.get_model_info(comp_id)}


class ModelsLoader(NodeBase):
    label = "Load Models"
    category = "loader"
    resizable = True
    params = {
        "repo_id": {
            "label": "Repository ID",
            "display": "modelselect",
            "type": "string",
            "default": {"source": "hub", "value": "OzzyGT/base-modular-loader"},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": ["StableDiffusionXLModularPipeline"]},
                    "local": {"className": ["StableDiffusionXLModularPipeline"]},
                },
            },
        },
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float16",
            "postProcess": str_to_dtype,
        },
        "device": {"label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "unet": {"label": "Unet", "display": "input", "type": "diffusers_auto_model"},
        "vae": {"label": "VAE", "display": "input", "type": "diffusers_auto_model"},
        "text_encoders": {"label": "Text Encoders", "display": "output", "type": "text_encoders"},
        "unet_out": {"label": "UNet", "display": "output", "type": "diffusers_auto_model"},
        "vae_out": {"label": "VAE", "display": "output", "type": "diffusers_auto_model"},
        "scheduler": {"label": "Scheduler", "display": "output", "type": "scheduler"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.loader = None

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        self.loader = None
        super().__del__()

    def execute(self, repo_id, device, dtype, unet=None, vae=None):
        logger.debug(f"""
            ModelsLoader ({self.node_id}) received parameters:
            - repo_id: {repo_id}
            - dtype: {dtype}
            - device: {device}
            - unet: {unet}
            - vae: {vae}
        """)

        components_to_update = {
            "unet": components.get_one(unet["model_id"]) if unet else None,
            "vae": components.get_one(vae["model_id"]) if vae else None,
        }
        components_to_update = {k: v for k, v in components_to_update.items() if v is not None}

        if isinstance(repo_id, dict):
            real_repo_id = repo_id.get("value", repo_id)
            source = repo_id.get("source", "hub")  # TODO: do something when is local?
        else:
            real_repo_id = ""

        self.loader = ModularPipeline.from_pretrained(
            real_repo_id, components_manager=components, collection=self.node_id
        )

        # YIYI/Alvaro TODO: do we need to limit to these components?
        ALL_COMPONENTS = ["unet", "vae", "scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
        components_to_load = [c for c in ALL_COMPONENTS if c not in components_to_update]
        components_to_reload = []

        for comp_name in components_to_load:
            comp_spec = self.loader.get_component_spec(comp_name)
            if comp_spec.load_id != "null":
                comp_with_same_load_id = components._lookup_ids(load_id=comp_spec.load_id)
                same_comp_in_collection = []
                for comp_id in comp_with_same_load_id:
                    if isinstance(components.get_one(component_id=comp_id), torch.nn.Module):
                        comp_dtype = components.get_one(component_id=comp_id).dtype
                        if comp_dtype == dtype:
                            same_comp_in_collection.append(comp_id)
                    else:
                        same_comp_in_collection.append(comp_id)
                if not same_comp_in_collection:
                    components_to_reload.append(comp_name)

        self.loader.load_components(names=components_to_reload, torch_dtype=dtype)
        self.loader.update_components(**components_to_update)

        print(f" ModelsLoader: reloaded components: {components_to_reload}")
        print(f" ModelsLoader: updated components: {components_to_update.keys()}")

        # Construct loaded_components at the end after all modifications
        loaded_components = {
            "unet_out": node_get_component_info(node_id=self.node_id, manager=components, name="unet"),
            "vae_out": node_get_component_info(node_id=self.node_id, manager=components, name="vae"),
            "text_encoders": {
                k: node_get_component_info(node_id=self.node_id, manager=components, name=k)
                for k in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
            },
            "scheduler": node_get_component_info(node_id=self.node_id, manager=components, name="scheduler"),
        }

        logger.debug(f" ModelsLoader: Final component_manager state: {components}")

        return loaded_components
