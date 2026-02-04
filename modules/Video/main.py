from mellon.NodeBase import NodeBase
from mellon.config import CONFIG
from pathlib import Path
import logging
from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from utils.paths import parse_filename

logger = logging.getLogger('mellon')

class Load(NodeBase):
    """
    Load a video from a file path.
    """

    label = "Load Video"
    category = "Video"
    resizable = True
    params = {
        "video": {
            "label": "Video",
            "display": "output",
            "type": "image",
        },
        'label': {
            "display": "ui_label",
            "value": "Load Video",
        },
        "file": {
            "label": False,
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {
                "fileTypes": ["video"],
                "multiple": False,
            },
        },
        "filename": { "label": "File Name", "display": "output", "type": "str" },
        "width": { "display": "output", "type": "int" },
        "height": { "display": "output", "type": "int" },
        "frames": { "display": "output", "type": "int" },
        "fps": { "label": "FPS", "display": "output", "type": "float" },
    }

    def execute(self, **kwargs):
        import imageio
        from PIL import Image
        file = kwargs["file"]
        file = Path(file[0] if isinstance(file, list) else file)
        logger.debug(f"Loading video from file: {file}")

        if file is None or file == "":
            file = ""
        if not Path(file).is_absolute():
            file = Path(CONFIG.paths['work_dir']) / file
        if not Path(file).exists():
            file = ""

        images = []
        reader = imageio.get_reader(str(file), 'ffmpeg')
        meta = reader.get_meta_data()
        width = meta.get('size', (0,0))[0]
        height = meta.get('size', (0,0))[1]
        frames = reader.count_frames()
        fps = meta.get('fps', 0)

        for frame in reader:
            images.append(Image.fromarray(frame))
        reader.close()

        return {
            "filename": str(file),
            "video": images,
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
        }

class Export(NodeBase):
    """
    Save/Re-encode a video
    """
    label = "Export Video"
    category = "video"
    resizable = True
    params = {
        "video": { "type": ["video", "str", "image"], "display": "input" },
        "filename": {
            "label": "File",
            "type": "str",
            "default": "{PATH:videos}/Mellon_{HASH:6}.mp4",
        },
        #"codec": { "type": "str", "options": ["libx264", "vp9"], "default": "libx264" },
        "quality": { "display": "slider", "type": "int", "min": 1, "max": 10, "default": 5 },
        "fps": { "label": "FPS", "type": "float", "default": 24, "min": 1, "max": 240, "step": 0.01 },
        "preview": { "display": "ui_video", "type": "url", "dataSource": "file" },
        "file": { "type": "video", "display": "output" },
    }

    def execute(self, **kwargs):
        import imageio
        import numpy as np
        from PIL import Image

        video = kwargs["video"]
        filename = kwargs.get("filename", "{PATH:videos}/Mellon_{HASH:6}.mp4")
        quality = kwargs.get("quality", 5)
        fps = kwargs.get("fps", 24)

        output = None
        parsed_filename = parse_filename(filename)

        Path(parsed_filename).parent.mkdir(parents=True, exist_ok=True)

        if isinstance(video, str):
            reader = imageio.get_reader(video)
            writer = imageio.get_writer(parsed_filename, 
                                        fps=fps, 
                                        quality=quality,
                                        codec='libx264',
                                        )
            for frame in reader:
                writer.append_data(frame)
            reader.close()
            writer.close()

        elif isinstance(video, list):
            # It's a list of images
            if not video:
                return {"file": None}
            
            if not isinstance(video, list):
                video = [video]

            if isinstance(video[0], Image.Image):
                video = [np.array(img) for img in video]

            writer = imageio.get_writer(parsed_filename, 
                                        fps=fps, 
                                        quality=quality,
                                        codec='libx264',
                                        )
            for frame in video:
                writer.append_data(frame)
            writer.close()

        return {
            "file": str(parsed_filename),
        }