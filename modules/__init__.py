import logging
import ast
from os import scandir, path
from importlib import import_module
from typing import Dict, Any, Set
from types import ModuleType
logger = logging.getLogger('mellon')
logger.debug("Loading modules...")

MODULE_MAP = {}
def safe_eval_ast_node_recursive(node: ast.AST, module_obj: ModuleType) -> Any:
    """
    Recursively evaluate an AST node, handling nested structures and resolving
    names (like functions) against a provided live module object.
    """
    try:
        # Try to evaluate as a simple literal first (numbers, strings, bools, None)
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        # If it's a dictionary, recursively evaluate its keys and values
        if isinstance(node, ast.Dict):
            result = {}
            for key_node, value_node in zip(node.keys, node.values):
                key = safe_eval_ast_node_recursive(key_node, module_obj)
                value = safe_eval_ast_node_recursive(value_node, module_obj)
                result[key] = value
            return result
        # If it's a list, recursively evaluate its elements
        elif isinstance(node, ast.List):
            return [safe_eval_ast_node_recursive(item, module_obj) for item in node.elts]
        # If it's a tuple, recursively evaluate its elements
        elif isinstance(node, ast.Tuple):
            return tuple(safe_eval_ast_node_recursive(item, module_obj) for item in node.elts)
        
        # --- KEY CHANGE ---
        # If it's a name (e.g., a variable or function name), try to resolve it.
        elif isinstance(node, ast.Name):
            try:
                # Attempt to get the attribute (the function/variable) from the live module
                return getattr(module_obj, node.id)
            except AttributeError:
                # If it fails, the name is not defined at the module level.
                # Fall back to returning the name as a string.
                logger.warning(f"Could not resolve name '{node.id}' in module '{module_obj.__name__}'. Storing as string.")
                return ast.unparse(node)
        
        # For other complex expressions (like function calls `my_func()`), convert to string
        else:
            return ast.unparse(node)

def parse_node_class(node: ast.ClassDef, module_obj: ModuleType) -> Dict[str, Any]:
    """
    Parse a NodeBase class and extract its module definition.
    Requires the live module object to resolve function references.
    """
    module_def = {
        "label": None,
        "category": None,
        "description": None,
        "resizable": None,
        "style": {},
        "params": {},
    }

    if ast.get_docstring(node):
        module_def["description"] = ast.get_docstring(node)

    for item in node.body:
        # Get class variables (e.g., label, category, params)
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id in module_def:
                    # Pass the module object to the evaluator to resolve names
                    module_def[target.id] = safe_eval_ast_node_recursive(item.value, module_obj)
    
    return module_def

def parse_module_file(module_filepath: str, module_obj: ModuleType, ignore_classes: Set[str] = None) -> Dict[str, Any]:
    """
    Parses a python file using AST and extracts a map of all classes that inherit from NodeBase.
    """
    if ignore_classes is None:
        ignore_classes = set()

    try:
        with open(module_filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=module_filepath)
        
        class_map = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name not in ignore_classes:
                # Check if the class inherits from NodeBase
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "NodeBase":
                        # Pass the module object to the class parser
                        class_map[node.name] = parse_node_class(node, module_obj)
                        break
        return class_map
    except Exception as e:
        logger.error(f"Failed to parse {module_filepath}: {e}", exc_info=True)
        return {}

def parse_module_map(base_path: str) -> None:
    """
    Scans a directory for modules, imports them, and parses them to build the MODULE_MAP.
    """
    for entry in scandir(base_path):
        if entry.is_dir() and not entry.name.startswith(("__", ".")):
            module_name = f"{base_path}.{entry.name}"
            try:
                # Import the module to get the live object for runtime introspection
                module_obj = import_module(module_name)
            except ImportError as e:
                logger.error(f"Failed to import module '{module_name}': {e}")
                continue

            # Check if the module provides a pre-computed map
            module_content = {}
            if hasattr(module_obj, "MODULE_MAP"):
                module_content = module_obj.MODULE_MAP.copy()
                logger.debug(f"Loaded {len(module_content)} classes from {module_name}")

            # Determine which files within the module to parse
            files_to_parse = getattr(module_obj, "MODULE_PARSE", ["main"])
            ignore_classes = set(module_content.keys())
            
            for file_stem in files_to_parse:
                # We need to find the full path of the file to parse
                # The module object's __file__ attribute usually points to the __init__.py
                module_dir = path.dirname(module_obj.__file__)
                target_file = path.join(module_dir, f"{file_stem}.py")

                if path.exists(target_file):
                    # Pass the file path and the live module object to the parser
                    parsed_content = parse_module_file(target_file, module_obj, ignore_classes)
                    if parsed_content:
                        logger.debug(f"Parsed {len(parsed_content)} classes from '{target_file}'")
                        module_content.update(parsed_content)
                else:
                    logger.warning(f"Could not find file '{file_stem}.py' in module '{module_name}'")

            if module_content:
                MODULE_MAP[module_name] = module_content
                logger.info(f"Loaded {len(module_content)} class{'' if len(module_content) == 1 else 'es'} from module '{module_name}'.")
            else:
                logger.warning(f"Module '{module_name}' could not be parsed or has no NodeBase classes.")

parse_module_map("modules")
parse_module_map("custom")

logger.info(f"Loaded {len(MODULE_MAP)} modules")
