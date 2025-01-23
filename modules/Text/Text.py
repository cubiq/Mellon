from mellon.NodeBase import NodeBase

class Text(NodeBase):
    def execute(self, text_input):
        return { 'text_output': text_input }

class Display(NodeBase):
    def execute(self, text_input):
        return { 'text_output': text_input }