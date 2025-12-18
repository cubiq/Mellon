import logging

import torch
from diffusers.modular_pipelines import ModularPipeline
from PIL import Image

from mellon.NodeBase import NodeBase
from .utils import collect_model_ids
from .modular_utils import pipeline_class_to_mellon_node_config, DummyCustomPipeline
import importlib
from . import components


logger = logging.getLogger("mellon")


#YiYi Notes: this is not working for qwen/flux as latents needs to be unpacked first
class LatentsPreview(NodeBase):
    label = "Latents Preview"
    category = "image"
    resizable = True
    params = {
        "latents": {"label": "Latents", "display": "input", "type": "latent"},
        "image": {"label": "Image", "display": "output", "type": "image", "hidden": True},
        "preview": {"display": "ui_image", "dataSource": "image"},
    }

    def execute(self, latents):
        latent_rgb_factors = [
            [0.3920, 0.4054, 0.4549],
            [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076],
        ]

        image = None

        if latents is not None:
            latent_rgb_factors = torch.tensor(latent_rgb_factors, dtype=latents.dtype).to(device=latents.device)
            latent_image = latents.squeeze(0).permute(1, 2, 0) @ latent_rgb_factors
            latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
            denoised_image = latents_ubyte.byte().cpu().numpy()
            image = Image.fromarray(denoised_image)
            image = image.resize((image.width * 2, image.height * 2), resample=Image.Resampling.BICUBIC)

        return {"image": image}


class DecodeLatents(NodeBase):
    label = "Decode Latents"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    node_type = "decoder"
    params = {
        "model_type": {
            "label": "Model Type",
            "type": "string",
            "default": "",
            "hidden": True,
        },
        "vae": {
            "label": "VAE *", 
            "display": "input", 
            "type": "diffusers_auto_model",
            "onSignal": [
                {
                    "action": "value",
                    "target": "model_type",
                    # "data": SIGNAL_DATA, # YiYi Notes: not working
                    "data": {
                        "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
                        "QwenImageModularPipeline": "QwenImageModularPipeline",
                        "QwenImageEditModularPipeline": "QwenImageEditModularPipeline",
                        "QwenImageEditPlusModularPipeline": "QwenImageEditPlusModularPipeline",
                        "FluxModularPipeline": "FluxModularPipeline",
                        "FluxKontextModularPipeline": "FluxKontextModularPipeline",
                        "DummyCustomPipeline": "DummyCustomPipeline",
                    },
                },
                {"action": "exec", "data": "update_node"},
            ]
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def update_node(self, values, ref):
        node_params = {
            "model_type": {
                "label": "Model Type",
                "type": "string",
                "default": "",
                "hidden": True,
            },
            "vae": {
                "label": "VAE *",
                "display": "input",
                "type": "diffusers_auto_model",
                "onSignal": [
                    {
                        "action": "value",
                        "target": "model_type",
                        "data": {
                            "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
                            "QwenImageModularPipeline": "QwenImageModularPipeline",
                            "QwenImageEditModularPipeline": "QwenImageEditModularPipeline",
                            "QwenImageEditPlusModularPipeline": "QwenImageEditPlusModularPipeline",
                            "FluxModularPipeline": "FluxModularPipeline",
                            "FluxKontextModularPipeline": "FluxKontextModularPipeline",
                            "DummyCustomPipeline": "DummyCustomPipeline",
                        },
                    },
                    {"action": "exec", "data": "update_node"},
                ],
            },
        }

        model_type = values.get("model_type", "")

        if model_type == "" or self._model_type == model_type:
            return None

        self._model_type = model_type
        if model_type == "DummyCustomPipeline":
            self._pipeline_class = DummyCustomPipeline
        else:
            diffusers_module = importlib.import_module("diffusers")
            self._pipeline_class = getattr(diffusers_module, model_type)

        _, node_config = pipeline_class_to_mellon_node_config(self._pipeline_class, self.node_type)

        if node_config is None:
            self.send_node_definition(node_params)
            return
        
        node_params_to_update = node_config["params"]
        node_params_to_update.pop("vae", None)
        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def execute(self, **kwargs):
        kwargs = dict(kwargs)

        # 1. Get node config
        blocks, node_config = pipeline_class_to_mellon_node_config(
            self._pipeline_class, self.node_type
        )

        # 2. Create pipeline
        repo_id = kwargs.get("vae")["repo_id"]
        self._pipeline = blocks.init_pipeline(repo_id, components_manager=components)

        # 3. Cast params to correct types (Mellon bug workaround)
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        # 4. Update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
            target_model_names=expected_component_names,
        )

        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)

        # 5. Compile runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            if isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
            elif name in blocks.input_names:
                node_kwargs[name] = value

        # 6. Run the pipeline
        node_output_state = self._pipeline(**node_kwargs)

        # 7. Prepare outputs based on node_config["output_names"]
        output_names = node_config["output_names"].copy()
        outputs = {}
        for name in output_names:
            if name == "doc":
                outputs["doc"] = self._pipeline.blocks.doc
            else:
                outputs[name] = node_output_state.get(name)

        return outputs


class ImageEncode(NodeBase):
    label = "Encode Image"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    node_type = "vae_encoder"
    params = {
        "model_type": {
            "label": "Model Type",
            "type": "string",
            "default": "",
            "hidden": True, # Hidden field to receive signal data
        },
        "vae": {
            "label": "VAE *",
            "display": "input",
            "type": "diffusers_auto_model",
            "onSignal": [
                {
                    "action": "value",
                    "target": "model_type",
                    "data": {
                        "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
                        "QwenImageModularPipeline": "QwenImageModularPipeline",
                        "QwenImageEditModularPipeline": "QwenImageEditModularPipeline",
                        "QwenImageEditPlusModularPipeline": "QwenImageEditPlusModularPipeline",
                        "FluxModularPipeline": "FluxModularPipeline",
                        "FluxKontextModularPipeline": "FluxKontextModularPipeline",
                        "DummyCustomPipeline": "DummyCustomPipeline",
                    },
                },
                {"action": "exec", "data": "update_node"},
            ],
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def update_node(self, values, ref):

        node_params = {
            "model_type": {
                "label": "Model Type",
                "type": "string",
                "default": "",
                "hidden": True,
            },
            "vae": {
                "label": "VAE *",
                "display": "input",
                "type": "diffusers_auto_model",
                "onSignal": [
                    {
                        "action": "value",
                        "target": "model_type",
                        "data": {
                            "StableDiffusionXLModularPipeline": "StableDiffusionXLModularPipeline",
                            "QwenImageModularPipeline": "QwenImageModularPipeline",
                            "QwenImageEditModularPipeline": "QwenImageEditModularPipeline",
                            "QwenImageEditPlusModularPipeline": "QwenImageEditPlusModularPipeline",
                            "FluxModularPipeline": "FluxModularPipeline",
                            "FluxKontextModularPipeline": "FluxKontextModularPipeline",
                            "DummyCustomPipeline": "DummyCustomPipeline",
                        },
                    },
                    {"action": "exec", "data": "update_node"},
                ],
            },
        }

        model_type = values.get("model_type", "")

        if model_type == "" or self._model_type == model_type:
            return None

        self._model_type = model_type
        if model_type == "DummyCustomPipeline":
            self._pipeline_class = DummyCustomPipeline
        else:
            diffusers_module = importlib.import_module("diffusers")
            self._pipeline_class = getattr(diffusers_module, model_type)

        _, node_config = pipeline_class_to_mellon_node_config(self._pipeline_class, self.node_type)

        if node_config is None:
            self.send_node_definition(node_params)
            return
        
        node_params_to_update = node_config["params"]
        node_params_to_update.pop("vae", None)

        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def execute(self, **kwargs):
        kwargs = dict(kwargs)

        # 1. Get node config
        blocks, node_config = pipeline_class_to_mellon_node_config(
            self._pipeline_class, self.node_type
        )

        # 2. Create pipeline
        repo_id = kwargs.get("vae")["repo_id"]
        self._pipeline = blocks.init_pipeline(repo_id, components_manager=components)

        # 3. Cast params to correct types (Mellon bug workaround)
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        # 4. Update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
            target_model_names=expected_component_names,
        )

        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)

        # 5. Compile runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            if isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
            elif name in blocks.input_names:
                node_kwargs[name] = value

        # 6. Run the pipeline
        node_output_state = self._pipeline(**node_kwargs)

        # 7. Prepare outputs based on node_config["output_names"]
        output_names = node_config["output_names"].copy()
        outputs = {}
        for name in output_names:
            if name == "doc":
                outputs["doc"] = self._pipeline.blocks.doc
            else:
                outputs[name] = node_output_state.get(name)

        return outputs
