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
