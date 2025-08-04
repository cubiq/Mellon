import logging
import ast
from os import scandir, path
from importlib import import_module
from typing import Dict, Any, Set
from types import ModuleType
logger = logging.getLogger('mellon')
logger.info("Loading modules...")

# preload common functions
from utils.huggingface import local_files_only, get_local_model_ids
from utils.torch_utils import str_to_dtype, DEVICE_LIST, DEFAULT_DEVICE, CPU_DEVICE, IS_CUDA
from mellon.modelstore import modelstore

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
            # Iterate over keys and values. A None key should be a dictionary unpacking (**).
            for key_node, value_node in zip(node.keys, node.values):
                if key_node is None:
                    # This is a dictionary unpacking (e.g., **other_params)
                    try:
                        unpacked_dict = safe_eval_ast_node_recursive(value_node, module_obj)
                        if isinstance(unpacked_dict, dict):
                            result.update(unpacked_dict)
                    except Exception as e:
                        logger.error(f"Error unpacking dictionary in {module_obj.__name__}: {e}", exc_info=True)
                else:
                    # This is a standard key-value pair
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
        # If it's a list comprehension, evaluate it
        elif isinstance(node, ast.ListComp):
            try:
                expr = ast.Expression(body=node)
                ast.fix_missing_locations(expr)
                
                # Compile the expression and then evaluate it in the context of the module
                code = compile(expr, filename="<ast>", mode="eval")
                return eval(code, module_obj.__dict__)
            except Exception as e:
                logger.error(f"Error evaluating list comprehension in module '{module_obj.__name__}': {e}", exc_info=True)
                return ast.unparse(node) # Fallback to string on error

        # If it's a name (e.g., a variable or function name), try to resolve it.
        elif isinstance(node, ast.Name):
            try:
                # First, try to get the attribute from the live module
                return getattr(module_obj, node.id)
            except AttributeError:
                # If not found in module, try to get it from the global scope
                try:
                    return globals()[node.id]
                except KeyError:
                    # If it fails, the name is not defined at the module level or global scope.
                    # Fall back to returning the name as a string.
                    logger.warning(f"Could not resolve name '{node.id}' in module '{module_obj.__name__}'. Storing as string.")
                    return ast.unparse(node)

        # If it's a function call, evaluate the function and its arguments
        elif isinstance(node, ast.Call):
            # 1. Resolve the callable (e.g., the function object)
            func = safe_eval_ast_node_recursive(node.func, module_obj)
            
            # 2. Check if it's actually a callable function
            if not callable(func):
                logger.warning(f"Resolved name '{getattr(node.func, 'id', 'unknown')}' is not callable. Storing as string.")
                return ast.unparse(node)

            # 3. Evaluate positional and keyword arguments
            args = [safe_eval_ast_node_recursive(arg, module_obj) for arg in node.args]
            kwargs = {
                kw.arg: safe_eval_ast_node_recursive(kw.value, module_obj)
                for kw in node.keywords
            }
            
            # 4. Execute the function and return its result
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error executing function '{getattr(node.func, 'id', 'unknown')}' in module '{module_obj.__name__}': {e}", exc_info=True)
                return ast.unparse(node) # Fallback to string on error

        # fallback for other complex expressions
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
        "skipParamsCheck": False,
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
    ignore_classes = ignore_classes or set()

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

total_nodes = 0
def parse_module_map(base_path: str) -> None:
    """
    Scans a directory for modules, imports them, and parses them to build the MODULE_MAP.
    """
    global total_nodes
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
                        module_content.update(parsed_content)
                else:
                    logger.warning(f"Could not find file '{file_stem}.py' in module '{module_name}'")

            if module_content:
                #MODULE_MAP[module_name] = dict(sorted(module_content.items(), key=lambda item: item[0]))
                MODULE_MAP[module_name] = module_content
                total_nodes += len(module_content)
                logger.debug(f"Loaded {len(module_content)} Node{'' if len(module_content) == 1 else 's'} from '{module_name.split('.', 1)[-1]}'.")
            else:
                logger.warning(f"Module '{module_name}' could not be parsed or has no NodeBase classes.")

parse_module_map("modules")
parse_module_map("custom")

logger.info(f"Loaded {total_nodes} nodes from {len(MODULE_MAP)} modules.")