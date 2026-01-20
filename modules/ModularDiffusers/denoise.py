import importlib
import logging
from typing import Any, List, Tuple

import torch
from diffusers import ComponentsManager
from diffusers.modular_pipelines import BlockState, InputParam, LoopSequentialPipelineBlocks, ModularPipelineBlocks

from mellon.NodeBase import NodeBase

from . import MESSAGE_DURATION, components
from .modular_utils import DummyCustomPipeline, pipeline_class_to_mellon_node_config
from .utils import collect_model_ids


logger = logging.getLogger("mellon")


class PreviewBlock(ModularPipelineBlocks):
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("callback", default=None),
        ]

    def __call__(self, components: ComponentsManager, block_state: BlockState, i: int, t: int):
        block_state.callback(block_state.latents, i, components.scheduler.order)

        return components, block_state


def insert_preview_block(pipeline):
    """Insert preview_block into all LoopSequentialPipelineBlocks with 'denoise' in name."""

    # new block so we can preview the generation
    preview_block = PreviewBlock()

    def insert_preview_block_recursive(blocks, blocks_name, preview_block):
        if hasattr(blocks, "sub_blocks"):
            if isinstance(blocks, LoopSequentialPipelineBlocks) and "denoise" in blocks_name.lower():
                blocks.sub_blocks.insert("preview_block", preview_block, len(blocks.sub_blocks))
            else:
                for sub_block_name, sub_block in blocks.sub_blocks.items():
                    insert_preview_block_recursive(sub_block, sub_block_name, preview_block)

    insert_preview_block_recursive(pipeline.blocks.sub_blocks["denoise"], "", preview_block)


# SIGNAL_DATA = get_model_type_signal_data()


class Denoise(NodeBase):
    label = "Denoise"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    node_type = "denoise"
    params = {
        "unet": {
            "label": "Denoise Model *",
            "display": "input",
            "type": "diffusers_auto_model",
            "onSignal": [
                "update_node",
                {"action": "signal", "target": "guider"},
                {"action": "signal", "target": "controlnet_bundle"},
            ],
        },
    }

    def update_node(self, values, ref):
        node_params = {}
        model_type = self.get_signal_value("unet")

        if self._model_type == model_type:
            return None

        if model_type is None or model_type == "" or model_type == "DummyCustomPipeline":
            self._pipeline_class = DummyCustomPipeline
        else:
            diffusers_module = importlib.import_module("diffusers")
            self._pipeline_class = getattr(diffusers_module, model_type)

        self._model_type = model_type

        _, node_config = pipeline_class_to_mellon_node_config(self._pipeline_class, self.node_type)
        # not support this node type
        if node_config is None:
            self.send_node_definition(node_params)
            return

        node_params_to_update = node_config["params"]
        node_params_to_update.pop("unet", None)

        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def execute(self, **kwargs):
        kwargs = dict(kwargs)
        # 1. Get node config
        blocks, node_config = pipeline_class_to_mellon_node_config(self._pipeline_class, self.node_type)

        # 2. create pipeline

        # get device and repo_id
        device = None
        repo_id = None

        if (unet := kwargs.get("unet")) and isinstance(unet, dict):
            device = unet.get("execution_device")
            repo_id = unet.get("repo_id")

        if device is None or repo_id is None:
            self.notify(
                "You have to connect the denoise model",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        self._pipeline = blocks.init_pipeline(repo_id, components_manager=components)

        # YiYi TODO: add the preview block
        # insert_preview_block(self._denoise_node)

        # def preview_callback(latents, step_index: int, scheduler_order: int):
        #     if (step_index + 1) % scheduler_order == 0:
        #         self.trigger_output("latents_preview", latents)
        #         progress = int((step_index + 1) / num_inference_steps * 100 / scheduler_order)
        #         self.progress(progress)

        # YiYi Notes: take an extra step to cast the params to the correct type.
        # This due to Mellon bugs, should not need to take this step.
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        # 3. update components
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

        # 4. compile a dict of runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            value = kwargs.get(name)
            if value is None:
                continue

            # special case #1: `seed` -> always create a `generator`
            if name == "seed":
                generator = torch.Generator(device=device).manual_seed(value)
                node_kwargs["generator"] = generator

            # special case #2: passed `guidance_scale` but pipeline does not accept it
            # -> potentially create a new guider if pipeline support it
            elif name == "guidance_scale" and "guidance_scale" not in blocks.input_names:
                if "guider" in self._pipeline.component_names and "guider" not in components_to_update:
                    guider_spec = self._pipeline.get_component_spec("guider")
                    guider = guider_spec.create(guidance_scale=value)
                    self._pipeline.update_components(guider=guider)

            # if a dict is passed and is not an pipeline input, we unpack and process its contents
            # e.g. `embeddings` from text_encoder node
            elif isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
                    else:
                        expected_inputs = "\n  - ".join(blocks.input_names)
                        logger.warning(
                            f"Input '{name}:{k}' is not expected by {self.node_type} blocks.\n"
                            f"Expected inputs:\n  - {expected_inputs} \n"
                            f"Blocks: {blocks}"
                        )
            # pass the value as it is to the pipeline
            else:
                node_kwargs[name] = value

        # YiYi Notes: workaround on mellon bug, height/width was hidden but values are still passed in.
        if "image_latents" in node_kwargs and node_kwargs["image_latents"] is not None:
            node_kwargs.pop("height", None)
            node_kwargs.pop("width", None)

        # 5. figure out the outputs to return based on node_config["output_names"]
        outputs = {}
        output_names = node_config["output_names"].copy()
        # "doc" is a standard node output but not a pipeline output
        if "doc" in output_names:
            output_names.remove("doc")
            outputs["doc"] = self._pipeline.blocks.doc

        # 6. run the pipeline and update the outputs dict with the pipeline outputs
        try:
            node_outputs = self._pipeline(**node_kwargs, output=output_names)
        except ValueError as e:
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            return None
        except AttributeError as e:
            # the config error should be the missing scheduler
            if "config" in str(e):
                self.notify(
                    "You have to connect the scheduler",
                    variant="error",
                    persist=False,
                    autoHideDuration=MESSAGE_DURATION,
                )
                return None

            # any other error just show the original message
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            return None

        outputs.update(node_outputs)

        return outputs
