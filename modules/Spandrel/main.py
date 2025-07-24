from os import path, makedirs
import logging
logger = logging.getLogger('mellon')

from mellon.NodeBase import NodeBase
from spandrel import ModelLoader
from utils.torch_utils import ImageToTensor, TensorToImage
from utils.image import resize
from mellon.config import CONFIG

class Upscaler(NodeBase):
    def execute(self, **kwargs):
        image = kwargs.get("image")
        model_id = kwargs.get("model_id", None)
        rescale = kwargs.get("rescale")
        device = kwargs.get("device")

        if not model_id:
            logger.error("Model ID is required")
            return {"output": None}

        model_id = model_id['path'] if isinstance(model_id, dict) and 'path' in model_id else model_id

        # check if model_id is a file
        if not path.isfile(model_id):
            from huggingface_hub import repo_exists, HfFileSystem, hf_hub_download
            hub_repo = repo_exists(model_id, token=CONFIG.hf['token'])

            if not hub_repo:
                logger.error(f"Model {model_id} not found on Hugging Face")
                return {"output": None}
            hffs = HfFileSystem()
            repo_files = hffs.ls(model_id, detail=False)
            # return the first file that ends with "safetensors", "pt", "pth", "ckpt" or "pkl"
            file = next((f for f in repo_files if f.endswith(('.safetensors', '.pt', '.pth', '.ckpt', '.pkl'))), None).split('/')[-1]

            if not file:
                logger.error(f"Model {model_id} not found on Hugging Face")
                return {"output": None}
            local_path = path.join(CONFIG.paths['upscalers'], model_id)
            # create the directory if it doesn't exist
            makedirs(local_path, exist_ok=True)
            # check if the file already exists
            if not path.isfile(path.join(local_path, file)):
                model_id = hf_hub_download(model_id, file, token=CONFIG.hf['token'], local_dir=local_path)
            else:
                model_id = path.join(local_path, file)

        try:
            model = ModelLoader().load_from_file(model_id).eval()
        except Exception as e:
            logger.error(f"Error loading Spandrel model {model_id}: {e}")
            raise e

        self.mm_add(model, priority=0)
        model = self.mm_load(model, device)
        output = self.mm_exec(lambda: self.upscale(image, model), device, exclude=[model])
        
        if rescale != 1.0:
            output = [resize(o, int(o.width * rescale), int(o.height * rescale), resample='LANCZOS') for o in output]

        return { "output": output }
    
    def upscale(self, image, model):
        image = ImageToTensor(image)
        image = image if isinstance(image, list) else [image]
        output = []

        for i in image:
            if i.ndim == 3:
                i = i.unsqueeze(0)
            # remove alpha channel
            if i.shape[1] == 4:
                i = i[:, :3]
            output.append(model(i.to(model.device)).to('cpu'))
        del image

        if output:
            output = TensorToImage(output)

        return output