
MODULE_MAP = {
    'MeshPreview': {
        'label': 'Mesh Preview',
        'category': '3D',
        'params': {
            'mesh': {
                'label': 'Mesh',
                'type': 'mesh',
                'display': 'input',
            },
            'preview': {
                'label': 'Preview',
                'display': 'ui',
                'source': 'glb_out',
                'type': '3d',
            },
            'glb_out': {
                'label': 'GLB model',
                'type': 'mesh',
                'display': 'output',
            },
        }
    },

    'MeshLoader': {
        'label': 'Load Mesh',
        'category': '3D',
        'params': {
            'path': {
                'label': 'Path',
                'type': 'string',
            },
            'mesh': {
                'label': 'Mesh',
                'type': 'mesh',
                'display': 'output',
            },
        }
    },

    'MeshSave': {
        'label': 'Save Mesh',
        'category': '3D',
        'params': {
            'mesh': {
                'label': 'Mesh',
                'type': 'mesh',
                'display': 'input',
            },
            'path': {
                'label': 'Path',
                'type': 'string',
            },
        }
    },

    'ReduceFaces': {
        'label': 'Reduce Faces',
        'category': '3D',
        'description': 'Use `meshing_decimation_quadric_edge_collapse` PyMeshLab filter to reduce the number of faces in the mesh',
        'params': {
            'mesh': {
                'label': 'Mesh',
                'type': 'mesh',
                'display': 'input',
            },
            'mesh_out': {
                'label': 'Mesh',
                'type': 'mesh',
                'display': 'output',
            },
            'method': {
                'label': 'Method',
                'type': 'string',
                'options': ['relative', 'absolute'],
                'default': 'absolute',
                'onChange': { 'action': 'show', 'target': { 'relative': 'target_percent', 'absolute': 'target_facenum'}},
            },
            'target_facenum': {
                'label': 'Target Facenum',
                'type': 'int',
                'min': 0,
                'default': 48000,
            },
            'target_percent': {
                'label': 'Target Percentage',
                'type': 'float',
                'default': 0.5,
                'display': 'slider',
                'min': 0,
                'max': 1,
                'step': 0.05,
            },
            'quality_thr': {
                'label': 'Quality Threshold',
                'description': 'The quality threshold for penalizing bad shaped faces. 0 means no penalty.',
                'type': 'float',
                'default': 0.5,
                'display': 'slider',
                'min': 0,
                'max': 1,
                'step': 0.05,
            },
            'preserve_boundary': {
                'label': 'Preserve Boundary',
                'description': 'The simplification process tries to not affect mesh boundaries.',
                'type': 'boolean',
                'default': True,
            },
            'boundary_weight': {
                'label': 'Boundary Weight',
                'description': 'The importance of the boundary during simplification. Higher weight means more likely to be preserved.',
                'type': 'float',
                'default': 2.0,
                'min': 0,
                'step': 0.1,
            },
            'preserve_topology': {
                'label': 'Preserve Topology',
                'type': 'boolean',
                'default': True,
            },
        }
    }

}
