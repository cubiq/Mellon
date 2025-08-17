from os import path, makedirs
import logging
logger = logging.getLogger('mellon')

from mellon.NodeBase import NodeBase
from spandrel import ModelLoader
from utils.torch_utils import ImageToTensor, TensorToImage
from utils.image import resize
from mellon.config import CONFIG

class Upscaler(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)

        self._model_path = None
        self._model = None

    def execute(self, **kwargs):
        image = kwargs.get("image")
        model_id = kwargs.get("model_id", None)
        model_source = model_id.get('source', 'local') if isinstance(model_id, dict) else 'local'
        model_path = model_id.get('value', None) if isinstance(model_id, dict) else model_id
        downscale = kwargs.get("downscale", 1.0)
        device = kwargs.get("device")
        online_status = CONFIG.hf['online_status']

        if not model_path:
            raise ValueError("Model ID is required")

        if model_source == 'hub':
            from utils.huggingface import cached_file_path, list_repo_models

            if model_path.endswith((".safetensors", ".pt", ".pth", ".ckpt", ".pkl", ".bin")):
                path_segments = model_path.split('/')
                if len(path_segments) < 3:
                    raise ValueError(f"Invalid model path: {model_path}")

                file = '/'.join(path_segments[2:])
                repo_id = '/'.join(path_segments[:2])
                model_path = cached_file_path(repo_id, file)
            else:
                file = next(iter(list_repo_models(repo_id=model_path, token=CONFIG.hf['token'])), None)
                repo_id = model_path
                model_path = cached_file_path(repo_id, file)

            if not model_path:
                if online_status == "Offline":
                    #logger.error(f"Model {model_path} not found in cache and online status is `offline`. Consider changing the online status or adding the model to the cache manually.")
                    raise FileNotFoundError(f"Model {model_path} not found in cache and online status is `offline`. Consider changing the online status or adding the model to the cache manually.")

                from huggingface_hub import hf_hub_download
                try:
                    model_path = hf_hub_download(repo_id, file, token=CONFIG.hf['token'])
                except Exception as e:
                    logger.error(f"Error downloading model {model_path}: {e}")
                    raise

        else:
            # check if path is absolute
            if not path.isabs(model_path):
                model_path = path.join(CONFIG.paths['models'], model_path)
            # check if file exist
            if not path.isfile(model_path):
                logger.error(f"Model {model_path} not found")
                return {"output": None}

        if not model_path:
            raise ValueError(f"Invalid model path: {model_path}")
    
        try:
            if model_path == self._model_path and self._model is not None:
                model = self._model
            else:
                model = ModelLoader().load_from_file(model_path).eval()
                self._model_path = model_path
                self._model = model
        except Exception as e:
            logger.error(f"Error loading Spandrel model {model_path}")
            raise e

        self.mm_add(model, priority=0)
        output = self.mm_exec(lambda: Upscaler.upscale(image, model), device, models=[model])

        if downscale != 1.0:
            output = [resize(o, int(o.width * downscale), int(o.height * downscale), resample='LANCZOS') for o in output]

        return { "output": output }

    @staticmethod
    def upscale(image, model):
        image = ImageToTensor(image)
        image = image if isinstance(image, list) else [image]
        output = []

        for img in image:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # remove alpha channel
            if img.shape[1] == 4:
                img = img[:, :3]

            output.append(model(img.to(model.device)).to('cpu'))
        del image

        if output:
            output = TensorToImage(output)

        return output