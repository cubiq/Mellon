from mellon.NodeBase import NodeBase

class CustomNode(NodeBase):
    def execute(self, **kwargs):
        rgb = kwargs.get("rgb", "0,0,0")
        rgb = [int(x) for x in rgb.split(",")]

        return { "output": rgb }