from mellon.NodeBase import NodeBase
from utils.torch_utils import str_to_dtype, DEVICE_LIST, DEFAULT_DEVICE
from utils.huggingface import local_files_only, get_local_model_ids
from huggingface_hub import snapshot_download
from mellon.config import CONFIG
from diffusers import FluxKontextPipeline, BitsAndBytesConfig, FluxTransformer2DModel

HF_TOKEN = CONFIG.hf['token']

class FluxPipelineLoader(NodeBase):
    label = "FLUX Kontext Pipeline Loader"
    category = "loader"
    params = {
        "pipeline": { "label": "FLUX Pipeline", "display": "output", "type": "pipeline" },
        "model_id": {
            "label": "Model",
            "display": "autocomplete",
            "type": "string",
            "default": "black-forest-labs/FLUX.1-Kontext-dev",
            "options": get_local_model_ids(class_name="FluxKontextPipeline"),
            "fieldOptions": { "noValidation": True, "model_loader": True }
        },
        "dtype": {
            "label": "Dtype",
            "type": "string",
            "default": "bfloat16",
            "options": ['auto', 'float32', 'float16', 'bfloat16'],
        },
    }
    def execute(self, **kwargs):
        dtype = str_to_dtype(kwargs['dtype'])
        model_id = kwargs.get('model_id', 'black-forest-labs/FLUX.1-Kontext-dev')
        if not local_files_only(model_id):
            # ignore the merge file
            snapshot_download(repo_id=model_id, token=HF_TOKEN, ignore_patterns=["flux1-kontext-dev.safetensors"])

        nf4_config = BitsAndBytesConfig(
            load_in_8bit=True
            #load_in_4bit=True,
            #bnb_4bit_quant_type="nf4",
            #bnb_4bit_compute_dtype=dtype,
        )
        nf4_transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=True,
            quantization_config=nf4_config,
        )

        pipeline = FluxKontextPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=HF_TOKEN,
            transformer=nf4_transformer,
            local_files_only=True,
        )

        self.mm_add(pipeline.vae, priority=2)
        self.mm_add(pipeline.text_encoder, priority=1)
        self.mm_add(pipeline.text_encoder_2, priority=1)
        self.mm_add(pipeline.transformer, priority=3)
        return { "pipeline": pipeline }
    

class FluxKontextSampler(NodeBase):
    label = "FLUX Kontext Sampler"
    category = "sampler"
    params = {
        "pipeline": { "label": "FLUX Pipeline", "display": "input", "type": "pipeline" },
        "image": { "label": "Image", "display": "input", "type": "image" },
        "latents": { "label": "Latents", "display": "output", "type": "image" },
        "prompt": { "label": "Prompt", "type": "text" },
        "cfg": { "label": "Guidance Scale", "type": "float", "default": 2.5 },

    }
    def execute(self, **kwargs):
        pipeline = kwargs['pipeline']
        prompt = kwargs['prompt']
        cfg = kwargs['cfg']
        image = kwargs['image']

        #pipeline.enable_model_cpu_offload()
        pipeline.to(DEFAULT_DEVICE)

        latents = pipeline(
            prompt=prompt,
            image=image,
            guidance_scale=cfg,
            #output_type="latent",
            callback_on_step_end=self.pipe_callback,
        ).images

        #latents = latents.to('cpu').detach().clone()

        return { "latents": latents }