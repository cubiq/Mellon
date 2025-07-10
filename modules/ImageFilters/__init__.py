from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE

MODULE_MAP = {
    "Canny": {
        "label": "Canny Edge Detection",
        "description": "Apply a Canny edge detection to an image.",
        "category": "image_filter",
        "style": { "width": "300px" },
        "params": {
            "image": { "label": "Image", "type": "image", "display": "input" },
            "low_threshold": { "label": "Low Threshold", "type": "float", "default": 0.1, "min": 0.01, "max": 1, "step": 0.01, "display": "slider" },
            "high_threshold": { "label": "High Threshold", "type": "float", "default": 0.2, "min": 0.01, "max": 1, "step": 0.01, "display": "slider" },
            "device": { "label": "Device", "type": "string", "default": 'cpu:0' if 'cpu:0' in DEVICE_LIST else DEFAULT_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Image", "type": "image", "display": "output" },
        }
    },

    "UnsharpMask": {
        "label": "Unsharp Mask",
        "description": "Apply an unsharp mask to an image. The value of the image×2 is subtracted from the blurred image.",
        "category": "image_filter",
        "style": { "width": "300px" },
        "params": {
            "image": { "label": "Image", "type": "image", "display": "input" },
            "radius": { "label": "Radius", "type": "float", "default": 5, "min": 1, "max": 63, "step": 1, "display": "slider", "description": "The radius of the blur applied to the image. Higher values will result in a sharper image." },
            "amount": { "label": "Amount", "type": "float", "default": 0.3, "min": 0.01, "max": 1, "step": 0.01, "display": "slider" },
            "device": { "label": "Device", "type": "string", "default": 'cpu:0' if 'cpu:0' in DEVICE_LIST else DEFAULT_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Image", "type": "image", "display": "output" },
        }
    }
}