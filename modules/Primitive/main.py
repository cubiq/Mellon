from mellon.NodeBase import NodeBase
from PIL import Image
import json

def serialize_exif_value(data, max_length=100):
    """Recursively serialize EXIF data values to be JSON compatible"""
    if isinstance(data, bytes):
        # Convert bytes to hex string for readability
        return f"{data.hex()[:50]}{'...' if len(data) > 50 else ''}"
    elif isinstance(data, (int, float, str, bool, type(None))):
        # These types are JSON serializable
        return data
    elif isinstance(data, dict):
        # Recursively handle dictionaries
        return {str(k): serialize_exif_value(v, max_length) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # Recursively handle lists and tuples
        return [serialize_exif_value(item, max_length) for item in data]
    else:
        # For other types, convert to string representation
        str_repr = str(data)
        if len(str_repr) > max_length:
            str_repr = str_repr[:max_length] + "..."
        return f"{str_repr}"

# A node that can preview the text value of most field types
class DataViewer(NodeBase):
    label = "Data Viewer"
    category = "primitive"
    resizable = True
    params = {
        "value": {
            "label": "Data",
            "display": "input",
            "type": "any",
        },
        "preview": {
            "label": "Preview",
            "display": "ui_text",
            "dataSource": "output",
        },
        "output": {
            "label": "Output",
            "display": "output",
            "type": "string",
        }
    }

    def execute(self, **kwargs):
        from PIL.ExifTags import TAGS

        value = kwargs["value"]
        if isinstance(value, dict) or isinstance(value, list):
            value = json.dumps(value, indent=2)
        elif isinstance(value, Image.Image):
            # Extract EXIF data
            exif_data = {}
            if hasattr(value, '_getexif') and value._getexif() is not None:
                exif = value._getexif()
                for tag_id in exif:
                    tag = TAGS.get(tag_id, tag_id)
                    data = exif.get(tag_id)
                    
                    # Use recursive serialization for all data types
                    exif_data[tag] = serialize_exif_value(data)
            
            value = json.dumps({
                "width": value.width,
                "height": value.height,
                "format": value.format,
                "mode": value.mode,
                "size": value.size,
                "filename": value.filename,
                "exif": exif_data
            }, indent=2)

        return {"output": str(value)}
    
class TextValue(NodeBase):
    label = "Text Value"
    category = "primitive"
    params = {
        "text": {
            "label": "Text",
            "display": "text",
            "type": "string",
        },
        "output": {
            "label": "Output",
            "display": "output",
            "type": "string",
        }
    }

    def execute(self, **kwargs):
        return {"output": kwargs["text"]}

class ToList(NodeBase):
    label = "Items to List"
    category = "primitive"
    params = {
        "item": { "type": "any", "display": "input", "spawn": True },
        "list": { "type": "any", "display": "output" },
    }
    
    def execute(self, **kwargs):
        return {"list": kwargs["item"]}
    

class DynamicNode(NodeBase):
    label = "Dynamic Node"
    category = "primitive"
    skipParamsCheck = True
    params = {
        "dropdown": { "type": "string", "options": ["option1", "option2", "option3"], "default": "option1", "onChange": "updateNode" },
        "text": { "type": "string", "display": "string", "label": "Text", "value": "This is a text field" },
        "button": {
            "display": "ui_button",
            "label": "Click me!",
            "onChange": "buttonAction",
            "fieldOptions": {
                "queue": True,            # queue the task to run in background, use for long running tasks
                #"variant": "outlined",    # button variant: contained, text, outlined
                #"color": "primary",       # button color: primary, secondary, error, info, success
            }
        },
    }

    def updateNode(self, values, ref):
        # To get the value of a field you can use params key
        value = values.get("dropdown")

        # To get the value of the calling field automatically, you can use the ref object
        key = ref.get("key")
        value_from_key = values.get(key) # this will be the value of "dropdown" because that's what called the updateNode function

        print(f"value: {value}, key: {key}, value_from_key: {value_from_key}")

        # All node values are passed in the values object
        print('All values', values)

        params = {}
        if value == "option1":
            params = {
                "string": { "label": "Field Option 1", "display": "string", "type": "string", "value": value },
            }
        elif value == "option2":
            params = {
                "toggle": { "label": "Toggle 2", "display": "toggle", "type": "boolean", "value": value },
                "string": { "label": "Field Option 2", "display": "string", "type": "string", "value": value },
            }
        elif value == "option3":
            params = {
                "text": { "label": "Text 3", "display": "text", "type": "string", "value": value },
                "select": { "label": "Select3", "display": "select", "type": "string", "options": ["option1", "option2", "option3"], "value": value },
                "string": { "label": "Field Option 3", "display": "string", "type": "string", "value": value },
            }
        
        self.send_node_definition(params)
    
    def buttonAction(self, values, ref):
        import time
        # simulate a long running task
        time.sleep(2)

        params = {
            "toggle1": { "label": "Toggle 1", "display": "toggle", "type": "boolean", "value": True },
            "toggle2": { "label": "Toggle 2", "display": "toggle", "type": "boolean", "value": False },
            "toggle3": { "label": "Toggle 3", "display": "toggle", "type": "boolean", "value": True },
            "text": { "label": "Text", "display": "text", "type": "string", "value": 'test' },
            "string": { "label": "Text field", "display": "string", "type": "string" },
            "input": { "label": "Input", "display": "input", "type": "string" },
            "output": { "label": "Output", "display": "output", "type": "string" },
        }
        self.send_node_definition(params)

    def execute(self, **kwargs):
        input = kwargs.get("input", "")
        text = kwargs.get("text", "")
        input = input if isinstance(input, list) else [input]
        
        output = {}
        if "output" in self.output:
            output = { "output": f"{', '.join(t for t in input)} - {text}" }

        return output