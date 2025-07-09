import os

def list_files(directory: str, recursive: bool = False, extensions: list[str] = [], relative_path: str = ""):
    if not os.path.exists(directory):
        return []

    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if extensions and not any(file.lower().endswith(f".{ext.lower()}") for ext in extensions):
                continue
            
            # Create the name with relative path structure
            if relative_path:
                name_with_path = f"{relative_path}/{file.split('.')[0]}"
            else:
                name_with_path = file.split(".")[0]
            
            files.append({
                "path": file_path,
                "name": file,
                "label": name_with_path,
                "extension": file.split(".")[-1] if "." in file else "",
            })
        elif recursive and os.path.isdir(file_path):
            new_relative_path = f"{relative_path}/{file}" if relative_path else file
            subdir_files = list_files(file_path, recursive, extensions, new_relative_path)
            files.extend(subdir_files)
        
        files.sort(key=lambda x: x["label"].lower())
    return files
