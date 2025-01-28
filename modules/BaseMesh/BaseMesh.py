from mellon.NodeBase import NodeBase

class MeshPreview(NodeBase):
    def execute(self, mesh):
        import trimesh
        if isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.export(file_type='glb')

        return { 'glb_out': mesh }

class MeshLoader(NodeBase):
    def execute(self, path):
        import trimesh
        mesh = trimesh.load_mesh(path)
        return { 'mesh': mesh }
    
class MeshSave(NodeBase):
    def execute(self, mesh, path):
        #import trimesh
        mesh.export(path)
        return { 'mesh': mesh }