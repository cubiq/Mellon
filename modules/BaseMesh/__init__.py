
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
    }


}
