import os
import re

def list_files(directory: str, recursive: bool = False, extensions: list[str] | tuple[str, ...] = [], match: str = "", relative_path: str = ""):
    if not os.path.exists(directory):
        return []

    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if extensions and not any(file.lower().endswith(f".{ext.lower()}") for ext in extensions):
                continue
            
            # Apply regex match filter if provided
            if match and not re.search(match, file_path):
                continue
            
            # Create the name with relative path structure
            name_with_path = f"{relative_path}/{file}" if relative_path else file

            files.append({
                "path": file_path,
                "rel_path": name_with_path,
                "name": file,
                "extension": file.split(".")[-1] if "." in file else "",
            })
        elif recursive and os.path.isdir(file_path):
            new_relative_path = f"{relative_path}/{file}" if relative_path else file
            subdir_files = list_files(file_path, recursive, extensions, match, new_relative_path)
            files.extend(subdir_files)
        
        files.sort(key=lambda x: x["rel_path"].lower())
    return files
