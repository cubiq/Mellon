import logging
logger = logging.getLogger('mellon')

from mellon.NodeBase import NodeBase
from spandrel import ModelLoader
from utils.torch_utils import ImageToTensor, TensorToImage
from utils.image import resize

class Upscaler(NodeBase):
    def execute(self, **kwargs):
        image = kwargs.get("image")
        model_id = kwargs.get("model_id")
        rescale = kwargs.get("rescale")
        device = kwargs.get("device")

        model_id = model_id['path'] if isinstance(model_id, dict) and 'path' in model_id else model_id

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