import logging
logger = logging.getLogger('mellon')
from huggingface_hub import scan_cache_dir, logging as hf_logging
hf_logging.set_verbosity_error()
from mellon.config import CONFIG
from pathlib import Path
import json
from typing import Optional, Callable
from mellon.config import CONFIG

def get_local_models(compact: bool = False):
    cache_dir = CONFIG.hf['cache_dir']
    try:
        cache = scan_cache_dir(cache_dir)
    except Exception as e:
        logger.error(f'Error scanning cache directory: {e}')
        return []
    local_models = []

    for repo in cache.repos:
        try:
            model = {
                'id': getattr(repo, 'repo_id', None),
                'type': getattr(repo, 'repo_type', None),
                'size': getattr(repo, 'size_on_disk', 0),
                'last_accessed': getattr(repo, 'last_accessed', None),
                'revisions': [],
                'class_names': [],
            }
            if model['id'] is None or model['type'] is None:
                logger.debug(f'Skipping invalid model: {repo}')
                continue

            if model['type'] == 'model' and hasattr(repo, 'revisions') and repo.revisions:
                for revision in repo.revisions:
                    rev = {
                        'hash': getattr(revision, 'commit_hash', None),
                        'size': getattr(revision, 'size_on_disk', 0),
                        'last_modified': getattr(revision, 'last_modified', None),
                    }
                    if rev['hash'] is None:
                        logger.debug(f'Skipping invalid revision: {revision}')
                        continue
                    model['revisions'].append(rev)

                last_revision = list(repo.revisions)[-1]
                for file in getattr(last_revision, 'files', []):
                    if getattr(file, 'file_name', None) and file.file_name.lower().endswith('.json'):
                        config = Path(getattr(file, 'file_path', None))
                        if config and config.exists():
                            try:
                                with open(config, 'r') as f:
                                    config_data = json.load(f)
                                if '_class_name' in config_data and config_data['_class_name'] not in model['class_names']:
                                    model['class_names'].append(config_data['_class_name'])
                            except (json.JSONDecodeError, IOError, TypeError) as e:
                                logger.debug(f'Error loading config file {config}: {e}')
                                continue

            if model['class_names']:
                model['class_names'].sort()

            if compact:
                model = {
                    'id': model['id'],
                    'class_names': model['class_names'],
                }

            local_models.append(model)
        except Exception as e:
            logger.debug(f'Error processing model {repo}: {e}')
            continue

    local_models.sort(key=lambda x: x['id'])
    return local_models

def get_local_model_ids(id: Optional[str] = None, class_name: Optional[str] | bool = None):
    local_models = get_local_models()
    if id:
        local_models = [model for model in local_models if id.lower() in model['id'].lower()]

    if class_name is not None:
        if isinstance(class_name, bool):
            local_models = [model for model in local_models if bool(model['class_names']) is class_name]
        else:
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