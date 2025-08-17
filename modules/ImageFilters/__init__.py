from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE, CPU_DEVICE

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
            "device": { "label": "Device", "type": "string", "default": CPU_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Image", "type": "image", "display": "output" },
        }
    },

    "UnsharpMask": {
        "label": "Unsharp Mask",
        "description": "Apply an unsharp mask to an image. A blurred version of the image is subtracted from the original image to enhance edges.",
        "category": "image_filter",
        "style": { "width": "300px" },
        "params": {
            "image": { "label": "Image", "type": "image", "display": "input" },
            "radius": { "label": "Radius", "type": "float", "default": 5, "min": 1, "max": 63, "step": 1, "display": "slider", "description": "The radius of the blur applied to the image. Higher values will result in a sharper image." },
            "amount": { "label": "Amount", "type": "float", "default": 0.3, "min": 0.01, "max": 1, "step": 0.01, "display": "slider" },
            "device": { "label": "Device", "type": "string", "default": CPU_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Image", "type": "image", "display": "output" },
        }
    },

    "GuidedBlur": {
        "label": "Guided Blur",
        "description": "Guided blur uses the original image to guide the blur process. The result preserves the edges while blurring other areas.",
        "category": "image_filter",
        "style": { "width": "300px" },
        "params": {
            "image": { "label": "Image", "type": "image", "display": "input" },
            "radius": { "label": "Radius", "type": "float", "default": 5, "min": 1, "max": 63, "step": 1, "display": "slider", "description": "The radius of the blur." },
            "eps": { "label": "Preserve Edges", "type": "float", "default": 0.02, "min": 0.01, "max": 0.2, "step": 0.01, "display": "slider", "description": "How strictly the filter preserves edges. Smaller values preserve more edges." },
            "subsample": { "label": "Subsample", "type": "int", "default": 1, "min": 1, "max": 16, "step": 1, "display": "slider", "description": "Subsample the image by this factor. Higher values are faster but less accurate." },
            "device": { "label": "Device", "type": "string", "default": CPU_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Image", "type": "image", "display": "output" },
        }
    },

    "AdaptiveSharpening": {
        "label": "Adaptive Sharpening",
        "description": "Contrast adaptive sharpening improves sharpness based on the image contrast, leaving smooth areas untouched.",
        "category": "image_filter",
        "style": { "width": "300px" },
        "params": {
            "image": { "label": "Image", "type": "image", "display": "input" },
            "sharpness": { "label": "Sharpness", "type": "float", "default": 0.8, "min": 0, "max": 1, "step": 0.1, "display": "slider" },
            "device": { "label": "Device", "type": "string", "default": CPU_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Image", "type": "image", "display": "output" },
        }
    },

    "GaussianBlur": {
        "label": "Gaussian Blur",
        "category": "image_filter",
        "style": { "width": "300px" },
        "params": {
            "image": { "label": "Image", "type": "image", "display": "input" },
            "amount": { "label": "Amount", "type": "float", "default": 5, "min": 1, "max": 100, "step": 1, "display": "slider" },
            "device": { "label": "Device", "type": "string", "default": CPU_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Image", "type": "image", "display": "output" },
        }
    }
}