MODULE_MAP = {
    'Text': {
        'label': 'Text',
        'description': 'Text',
        'category': 'text',
        'params': {
            'text_output': {
                'label': 'Text',
                'display': 'output',
                'type': 'string',
            },
            'text_input': {
                'label': 'Text',
                'type': 'string',
            },
        },
    },
    'Display': {
        'label': 'Display',
        'description': 'Display',
        'category': 'text',
        'params': {
            'text_output': {
                'label': 'Text',
                'display': 'output',
                'type': 'string',
            },
            'preview': {
                'label': 'Preview',
                'display': 'ui',
                'source': 'text_output',
                'type': 'string',
            },            
            'text_input': {
                'label': 'Text',
                'display': 'input',
                'type': 'string',
            },
        },
    },
}