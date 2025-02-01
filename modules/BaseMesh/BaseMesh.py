from mellon.NodeBase import NodeBase

# The following two functions are used to convert between pymeshlab and trimesh
# TODO: They seem robust but I'm sure they can be improved
def pymeshlab2trimesh(mesh):
    import trimesh
    import os
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=True)
    temp_file.close()
    temp_file_name = temp_file.name
    
    mesh.save_current_mesh(temp_file_name)
    mesh = trimesh.load(temp_file_name)
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)
          
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh

    return mesh

def trimesh2pymeshlab(mesh):
    import pymeshlab
    import trimesh
    import os
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=True)
    temp_file.close()
    temp_file_name = temp_file.name
    
    if isinstance(mesh, trimesh.scene.Scene):
        for idx, obj in enumerate(mesh.geometry.values()):
            if idx == 0:
                temp_mesh = obj
            else:
                temp_mesh = temp_mesh + obj
        mesh = temp_mesh
    mesh.export(temp_file_name)
    mesh = pymeshlab.MeshSet()
    mesh.load_new_mesh(temp_file_name)
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    return mesh


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
        mesh.export(path)
        return { 'mesh': mesh }

class ReduceFaces(NodeBase):
    def execute(
            self,
            mesh,
            method,
            target_facenum,
            target_percent,
            quality_thr,
            preserve_boundary,
            boundary_weight,
            preserve_topology):
        
        import trimesh
        if isinstance(mesh, trimesh.Trimesh):
            ms = trimesh2pymeshlab(mesh)
        else:
            ms = mesh

        cfg = {
            "qualitythr": quality_thr,
            "preserveboundary": preserve_boundary,
            "boundaryweight": boundary_weight,
            "preservetopology": preserve_topology,
            "preservenormal": True,
            "autoclean": True
        }
        
        if method == "absolute":
            cfg["targetfacenum"] = target_facenum
        elif method == "relative":
            cfg["targetperc"] = target_percent

        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse", **cfg
        )

        mesh_out = pymeshlab2trimesh(ms)
        ms.clear()
        del ms

        return { 'mesh_out': mesh_out }
