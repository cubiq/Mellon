import os

from mellon.NodeBase import NodeBase


class Lora(NodeBase):
    label = "Lora"
    category = "adapters"
    resizable = True
    params = {
        "model": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": [""]},
                    "local": {"className": [""]},
                },
            },
        },
        "weight_name": {
            "label": "Weight Name",
            "type": "string",
        },
        "scale": {
            "label": "Scale",
            "type": "float",
            "display": "slider",
            "default": 1.0,
            "min": -20,
            "max": 20,
            "step": 0.1,
        },
        "lora": {
            "label": "Lora",
            "type": "custom_lora",
            "display": "output",
        },
    }

    def execute(self, model, scale, weight_name=None):
        if isinstance(model, dict):
            lora_path = model.get("value", None)
            filename = os.path.splitext(os.path.basename(lora_path))[0]
        else:
            lora_path = None
            filename = ""

        adapter_name = f"{filename}_{self.node_id}"

        # Return the lora configuration directly, not wrapped in another dict
        return {
            "lora": {
                "lora_path": lora_path,
                "weight_name": weight_name,
                "adapter_name": adapter_name,
                "scale": scale,
            }
        }


class IPAdapter(NodeBase):
    label = "IP Adapter"
    category = "adapters"
    resizable = True

    params = {
        "model_path": {
            "label": "Model ID",
            "display": "modelselect",
            "type": "string",
            "default": {"source": "hub", "value": "h94/IP-Adapter"},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
            },
        },
        "subfolder": {"label": "Subfolder", "type": "string", "default": "sdxl_models"},
        "weight_name": {"label": "Weight Name", "type": "string", "default": "ip-adapter_sdxl_vit-h.safetensors"},
        "scale": {"label": "Scale", "type": "float", "default": 1.0, "min": 0, "max": 1, "step": 0.01},
        "image_encoder": {"label": "Image Encoder", "type": "diffusers_auto_model", "display": "input"},
        "image": {"label": "Image", "display": "input", "type": "image"},
        "ip_adapter": {"label": "IP-Adapter", "display": "output", "type": "custom_ip_adapter"},
    }

    def execute(self, model_path, weight_name, scale, image_encoder, image, subfolder=None):
        subfolder = None if subfolder == "" else subfolder

        # TODO: here ideally we should use the image_encoder and return the embeddings only

        if isinstance(model_path, dict):
            model_id = model_path.get("value", None)
        else:
            model_id = None

        return {
            "ip_adapter": {
                "model_id": model_id,
                "weight_name": weight_name,
                "subfolder": subfolder,
                "scale": scale,
                "image_encoder": image_encoder,
                "image": image,
            }
        }
