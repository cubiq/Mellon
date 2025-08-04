from utils.huggingface import get_local_models as hf_get_local_models, cached_file_path
from utils.paths import list_files
from mellon.config import CONFIG
import re

LOCAL_MODELS_PATH = CONFIG.paths['models']
ONLINE_STATUS = CONFIG.hf['online_status']
MODEL_EXTENSIONS = ("safetensors", "pt", "pth", "ckpt", "pkl", "bin")

class ModelStore:
    def __init__(self):
        self.hf = {}
        self.local = {}
        self.actualize()

    def update_hf(self):
        self.hf = hf_get_local_models()

    def update_local(self):
        self.local = list_files(LOCAL_MODELS_PATH, True, MODEL_EXTENSIONS)

    def actualize(self):
        self.update_hf()
        self.update_local()

    def get_hf_models(self, id: list[str] | str = None, class_name: list[str] | str = None, return_type: str = "full"):
        output = self.hf

        id = id or []
        class_name = class_name or []

        if isinstance(id, list) and id:
            try:
                output = [model for model in output if any(str(i).lower() == str(model['id']).lower() for i in id)]
            except Exception:
                output = self.hf
        elif isinstance(id, str):
            # if string perform a regex match
            output = [model for model in output if re.search(id, model['id'], re.IGNORECASE)]

        if isinstance(class_name, list) and class_name:
            try:
                output = [model for model in output if any(str(i).lower() in model['class_names'] for i in class_name)]
            except Exception:
                pass
        elif isinstance(class_name, str):
            # if string perform a regex match on each of the class names
            try:
                output = [model for model in output if any(re.search(class_name, cn, re.IGNORECASE) for cn in model['class_names'])]
            except Exception:
                pass

        if return_type == "ids":
            return [model['id'] for model in output]
        elif return_type == "compact":
            return [{'id': model['id'], 'class_names': model['class_names']} for model in output]

        return output

    def get_hf_ids(self, id: list[str] | str = None, class_name: list[str] | str = None):
        return self.get_hf_models(id=id, class_name=class_name, return_type="ids")

    def get_local_models(self, name: list[str] | str = None, return_type: str = "full"):
        output = self.local

        name = name or []

        if isinstance(name, list) and name:
            try:
                output = [model for model in output if any(str(i).lower() == str(model['rel_path']).lower() for i in name)]
            except Exception:
                output = self.local
        elif isinstance(name, str):
            # if string perform a regex match
            try:
                output = [model for model in output if re.search(name, model['rel_path'], re.IGNORECASE)]
            except Exception:
                pass

        if return_type == "ids" or return_type == "rel_path":
            return [model['rel_path'] for model in output]
        elif return_type == "path" or return_type == "full_path":
            return [model['path'] for model in output]

        return output

    def get_local_ids(self, name: list[str] | str = None):
        return self.get_local_models(name=name, return_type="ids")

    def is_hf_cached(self, id: str, actualize: bool = False):
        if actualize:
            self.update_hf()
        return any(model['id'] == id for model in self.hf)

    def is_local_cached(self, name: str, actualize: bool = False):
        if actualize:
            self.update_local()
        return any(model['rel_path'] == name for model in self.local)

    def is_cached(self, id: str, actualize: bool = False):
        if actualize:
            self.actualize()

        if self.is_hf_cached(id):
            return 'hf'
        if self.is_local_cached(id):
            return 'local'

        return False
    
    def offline_mode(self, id: str):
        if id is None:
            return False

        if ONLINE_STATUS == 'Offline':
            return True
        if ONLINE_STATUS == 'Online':
            return False

        if len(id.strip('/').split('/')) > 2:
            id = id.strip('/').split('/')
            file = '/'.join(id[2:])
            id = '/'.join(id[:2])
            return bool(cached_file_path(id, file))

        return bool(self.is_cached(id))

# Initialize the model store
modelstore = ModelStore()
