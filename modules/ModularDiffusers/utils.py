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
