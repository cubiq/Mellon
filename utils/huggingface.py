import logging
logger = logging.getLogger('mellon')
from huggingface_hub import scan_cache_dir, logging as hf_logging
hf_logging.set_verbosity_error()
from mellon.config import CONFIG
from pathlib import Path
import json
from typing import Optional, Callable
from mellon.config import CONFIG

def get_local_models():
    cache_dir = CONFIG.hf['cache_dir']
    cache = scan_cache_dir(cache_dir)
    local_models = []

    for repo in cache.repos:
        model = {
            'id': repo.repo_id,
            'type': repo.repo_type,
            'size': repo.size_on_disk,
            'last_accessed': repo.last_accessed,
            'revisions': [],
            'class_names': [],
        }

        if repo.repo_type == 'model' and repo.revisions:
            for revision in repo.revisions:
                model['revisions'].append({
                    'hash': revision.commit_hash,
                    'size': revision.size_on_disk,
                    'last_modified': revision.last_modified,
                })

            last_revision = list(repo.revisions)[-1]
            for file in last_revision.files:
                if file.file_name.lower().endswith('.json'):
                    config = Path(file.file_path)
                    if config.exists():
                        with open(config, 'r') as f:
                            config_data = json.load(f)
                        if '_class_name' in config_data and config_data['_class_name'] not in model['class_names']:
                            model['class_names'].append(config_data['_class_name'])

        if model['class_names']:
            model['class_names'].sort()

        local_models.append(model)

    local_models.sort(key=lambda x: x['id'])

    return local_models

def get_local_model_ids(id: Optional[str] = None, class_name: Optional[str] | bool = None):
    local_models = get_local_models()
    if id:
        local_models = [model for model in local_models if id.lower() in model['id'].lower()]
    if class_name is not None:
        if class_name == True: # all models with class names
            local_models = [model for model in local_models if model['class_names']]
        elif class_name == False: # all models without class names
            local_models = [model for model in local_models if len(model['class_names']) == 0]
        else: # specific class name
            local_models = [model for model in local_models if class_name in model['class_names']]

    return [model['id'] for model in local_models]

def delete_model(*revisions: str):
    cache_dir = CONFIG.hf['cache_dir']
    cache = scan_cache_dir(cache_dir)

    strategy = cache.delete_revisions(*revisions)

    if not strategy.repos:
        logger.error(f'No models to delete')
        return False

    try:
        strategy.execute()
        logger.info(f'Deleted {len(strategy.repos)} HF models, freed {strategy.expected_freed_size/1024**3:.2f} GB.')
        return True
    except Exception as e:
        logger.error(f'Error deleting HF models: {e}')
        return False

def search_hub(query: str, limit: int = 100):
    from huggingface_hub import HfApi
    api = HfApi(token=CONFIG.hf['token'], library_name='Mellon')
    results = api.list_models(search=query, limit=limit)
    models = []
    for result in results:
        models.append({
            'id': result.modelId,
            'created_at': result.created_at.timestamp() if result.created_at else None,
            'last_modified': result.last_modified.timestamp() if result.last_modified else None,
            'private': result.private,
            'downloads': result.downloads,
            'likes': result.likes,
            'gated': result.gated,
            #'library_name': result.library_name,
            'tags': result.tags,
            'pipeline_tag': result.pipeline_tag,
        })

    return models

# TODO: not yet implemented
def download_hub_model(model_id: str, progress_cb: Optional[Callable[[float], None]] = None):
    from huggingface_hub import snapshot_download
    
    cache_dir = CONFIG.hf['cache_dir']
    token = CONFIG.hf['token']

    try:
        snapshot_download(
            repo_id=model_id, 
            cache_dir=cache_dir, 
            token=token, 
            tqdm_class=progress_cb
        )
        if progress_cb:
            progress_cb(1.0)

    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {e}")
        return False

    return True

def local_files_only(model_id: str):
    online_status = CONFIG.hf['online_status']
    return online_status == 'Offline' or (online_status == 'Auto' and model_id in get_local_models())