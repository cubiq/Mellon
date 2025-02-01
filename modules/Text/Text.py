from mellon.NodeBase import NodeBase

class Text(NodeBase):
    def execute(self, text_field):
        return text_field

class DisplayText(NodeBase):
    def execute(self, text_in):
        return text_in