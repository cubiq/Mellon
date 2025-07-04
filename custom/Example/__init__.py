MODULE_MAP = {
    "CustomNode": {
        "description": "This is a custom node also showcasing a custom field.",
        "label": "Custom Node",
        "category": "primitive",
        "params": {
            "rgb": {
                "label": "Custom field: RGB Value",
                "display": "custom.ExampleField",
                "type": "string",
                "default": "0,0,0",
            },
            "output": {
                "label": "RGB Value",
                "display": "output",
                "type": "int",
            },
        },
    },
}