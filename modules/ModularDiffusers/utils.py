def combine_multi_inputs(inputs):
    """
    Recursively combines a list of dictionaries into a dictionary of lists.
    Handles nested dictionaries by recursively combining their values.

    Args:
        inputs: List of dictionaries to combine

    Returns:
        Dictionary where each key maps to either a list of values or a nested dictionary
        of combined values

    Example:
        inputs = [
            {
                "config": {"repo_id": "repo1", "subfolder": "unet"},
                "scale": 0.5
            },
            {
                "config": {"repo_id": "repo2", "subfolder": "vae"},
                "scale": 0.7
            }
        ]
        result = {
            "config": {
                "repo_id": ["repo1", "repo2"],
                "subfolder": ["unet", "vae"]
            },
            "scale": [0.5, 0.7]
        }
    """
    if not inputs:
        return {}

    # Get all unique keys from all dictionaries
    all_keys = set()
    for d in inputs:
        all_keys.update(d.keys())

    # Initialize the result dictionary
    result = {}

    # Process each key
    for key in all_keys:
        # Get all values for this key
        values = [d.get(key) for d in inputs]

        # If all values are dictionaries, recursively combine them
        if all(isinstance(v, dict) for v in values if v is not None):
            nested_values = [v for v in values if v is not None]
            if nested_values:  # Only combine if there are non-None values
                result[key] = combine_multi_inputs(nested_values)
        else:
            # For non-dictionary values, store as a list
            if any(v is not None for v in values):  # Only include if at least one non-None value
                result[key] = values

    return result


def collect_model_ids(kwargs, target_key_names, target_model_names=None):
    """
    Collect model_ids from kwargs for specified keys.
    Handles both flat (unet) and nested (text_encoders) structures.
    
    Args:
        kwargs: Dict of kwargs from Mellon node
        target_key_names: List of mellon param names to look into (e.g., ["unet", "text_encoders"])
        target_model_names: Optional list of component names to collect (e.g., ["transformer", "text_encoder", "tokenizer"]).
                           If None, collects all model_ids found.
    
    Returns:
        List of model_id strings
    
    Examples:
        >>> kwargs = {
        ...     "unet": {
        ...         "model_id": "transformer_140535472389568",
        ...         "execution_device": "cuda:0",
        ...         "repo_id": "Qwen/Qwen-Image",
        ...     },
        ...     "text_encoders": {
        ...         "text_encoder": {
        ...             "model_id": "text_encoder_140535474582288",
        ...             "execution_device": "cuda:0",
        ...         },
        ...         "tokenizer": {
        ...             "model_id": "tokenizer_140535596468592",
        ...         },
        ...         "repo_id": "Qwen/Qwen-Image",
        ...     },
        ...     "prompt": "a cat",
        ... }
        >>> collect_model_ids(
        ...     kwargs, 
        ...     target_key_names=["unet", "text_encoders"],
        ...     target_model_names=["transformer", "text_encoder"]  # excludes tokenizer
        ... )
        ["transformer_140535472389568", "text_encoder_140535474582288"]
    """
    
    def _collect_model_ids_recursive(data, filter_keys=None):
        """Recursively collect model_ids, optionally filtering by key names."""
        model_ids = []
        
        if not isinstance(data, dict):
            return model_ids
        
        # Flat structure: model_id at top level
        if "model_id" in data:
            model_ids.append(data["model_id"])
            return model_ids
        
        # Nested structure: recurse into sub-dicts
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            
            # If filter_keys specified, only process matching keys
            if filter_keys is not None and key not in filter_keys:
                continue
            
            model_ids.extend(_collect_model_ids_recursive(value, filter_keys))
        
        return model_ids
    
    model_ids = []
    
    for key, value in kwargs.items():
        if key in target_key_names and isinstance(value, dict):
            model_ids.extend(_collect_model_ids_recursive(value, target_model_names))
    
    return model_ids