from mellon.NodeBase import NodeBase

class TextToList(NodeBase):
    """
    Convert a text string into a list of strings
    """

    label = "Text to List"
    category = "text"
    resizable = True
    params = {
        "text": {"type": "string", "default": "", "display": "text"},
        "separator": {
            "type": "string",
            "options": {
                "\n": "\\n",
                ",": ",",
                " ": "[space]",
                "|": "|",
                ";": ";",
                ":": ":", 
            },
            "default": ","
        },
        "output": {"type": "string", "display": "output"},
    }

    def execute(self, **kwargs):
        text = kwargs["text"]
        separator = kwargs["separator"]
        print(separator)
        return [item.strip() for item in text.split(separator)]