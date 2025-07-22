import logging
logger = logging.getLogger('mellon')
import asyncio
from functools import partial
from importlib import import_module
from aiohttp import web, WSMsgType
from aiohttp_cors import setup as cors_setup, ResourceOptions
import json
import nanoid
from pathlib import Path
import traceback
import time
from copy import deepcopy
import hashlib

from mellon.config import CONFIG
from modules import MODULE_MAP
from utils.huggingface import get_local_models, delete_model, search_hub, download_hub_model
from utils.torch_utils import reset_memory_stats, get_memory_stats

class WebServer:
    def __init__(
            self,
            modules: dict = {},
            host: str = '127.0.0.1',
            port: int = 8088,
            secure: bool = False,
            certfile: str = None,
            keyfile: str = None,
            cors: bool = False,
            cors_routes: list = [],
            client_max_size: int = 1024**4,
            work_dir: str = 'data',
            data_dir: str = 'data'
        ):
        self.instance = nanoid.generate(size=10)

        self.modules = modules
        self.ws_sessions = {}

        self.interrupt_flag = False
        self.node_cache = {}

        self.queued_tasks = {}
        self.current_task = {}
        
        self.main_queue = asyncio.Queue()
        self.background_queue = asyncio.Queue()

        self.main_worker_task = None
        self.background_worker_task = None
        self.runner = None
        self.site = None

        self.host = host
        self.port = port
        self.ssl_context = None
        if secure and certfile and keyfile:
            import ssl
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile, keyfile)

        self.client_max_size = client_max_size
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.app = web.Application(client_max_size=self.client_max_size)

        # set up the routes
        self.app.add_routes([
            web.static('/assets', 'web/assets', append_version=True),
            web.static('/user', 'web/user', append_version=True),
            web.get('/', self.index),
            web.get('/favicon.ico', self.favicon),
            web.get('/ws', self.websocket),
            web.get(r'/nodes{id:/?([\w\d_-]+/[\w\d_-]+)?}', self.nodes),
            web.post('/fields/action', self.field_action),
            web.get('/cache/{node}/{field}', self.cache),
            web.get('/cache/{node}/{field}/{index}', self.cache),
            web.delete('/cache', self.delete_cache),
            web.get('/listdir', self.listdir),
            web.post('/file', self.filePost),
            web.get('/preview', self.preview),
            web.post('/graph', self.graph),
            web.get('/queue', self.get_queue),
            web.delete('/queue/{task_id}', self.delete_task),
            web.get('/stop', self.stop_execution),
            web.get('/hf_cache', self.hf_cache),
            web.delete('/hf_cache/{hash}', self.hf_cache_delete),
            web.get('/hf_hub', self.hf_hub),
            web.get('/hf_download', self.hf_download),
            web.get('/static/{module}/{file}', self.user_assets),
        ])

        # set up the cors routes
        if cors:
            cors = cors_setup(self.app, defaults={
                cors_route: ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")
                for cors_route in cors_routes
            })
            for route in list(self.app.router.routes()):
                cors.add(route)
    
    async def run(self):
        # Get the current event loop
        self.loop = asyncio.get_event_loop()
        
        # Start both workers
        self.main_worker_task = self.loop.create_task(self._main_worker())
        self.background_worker_task = self.loop.create_task(self._background_worker())
        
        self.runner = web.AppRunner(self.app, verbose=True, ssl=False)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host=self.host, port=self.port, ssl_context=self.ssl_context)
        await self.site.start()
    
    async def cleanup(self):
        # Stop the web server from accepting new connections.
        if self.site:
            await self.site.stop()

        # Close all active websocket connections.
        if self.ws_sessions:
            close_coroutines = [ws.close() for ws in self.ws_sessions.values()]
            await asyncio.gather(*close_coroutines, return_exceptions=True)

        # Cleanup the runner.
        if self.runner:
            await self.runner.cleanup()

        # Cancel and clean up the background workers.
        if self.main_worker_task:
            self.main_worker_task.cancel()
        if self.background_worker_task:
            self.background_worker_task.cancel()
        
        # Wait for the workers to finish.
        tasks_to_wait = [task for task in [self.main_worker_task, self.background_worker_task] if task]
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    """
    ╭───────────────╮
          Queue    
    ╰───────────────╯
    """
    def _get_queue(self):
        #task_list_sorted = {k: v for k, v in sorted(self.queued_tasks.items(), key=lambda x: x[1]['queued_at'], reverse=True)}
        # filter out keys that are not needed for the client
        queued_tasks = {
            k: {
                'name': v['name'],
                'sid': v['sid'],
                'queued_at': v['queued_at'],
                'task_id': k,
            }
            for k, v in self.queued_tasks.items()
        }

        current_task = None
        if self.current_task:
            current_task = {
                "task_id": self.current_task["task_id"],
                "name": self.current_task["name"],
                "sid": self.current_task["sid"],
                "started_at": self.current_task["started_at"],
                "progress": self.current_task["progress"],
            }
        
        return queued_tasks, current_task

    async def queue_task(self, task, args, future, sid, name=None):
        task_id = nanoid.generate(size=12)
        task_name = name or f'Unnamed task ({task.__name__})'

        self.queued_tasks[task_id] = {
            'task': task,
            'args': args,
            'future': future,
            "sid": sid,
            "queued_at": time.time(),
            "name": task_name,
        }
        await self.main_queue.put((task, args, future, task_id))

        task_list, current_task = self._get_queue()

        self.queue_message({
            "type": "task_queued",
            "task_id": task_id,
            "sid": sid,
            "queued": task_list,
            "current": current_task,
        })
        
    async def get_queue(self, _):
        """
        HTTP endpoint to return the tasks queue and the current task.
        """
        task_list, current_task = self._get_queue()
        return web.json_response({
            "queued": task_list,
            "current": current_task,
        })
    
    async def delete_task(self, request):
        """
        HTTP endpoint to delete a task from the queue.
        """
        task_id = request.match_info.get('task_id')
        if task_id in self.queued_tasks:
            task = self.queued_tasks.pop(task_id)
            logger.info(f"Task {task_id} {task['name']} deleted from queue.")
            task_list, current_task = self._get_queue()
            self.queue_message({
                "type": "task_cancelled",
                "task_id": task_id,
                "queued": task_list,
                "current": current_task,
            })
            return web.json_response({"error": False, "task_id": task_id, "queued": task_list, "current": current_task})
        elif self.current_task and self.current_task["task_id"] == task_id:
            return web.json_response({"error": True, "message": f"Task is already running and cannot be cancelled.", "task_id": task_id}, status=400)
        
        return web.json_response({"error": True, "message": f"Task not found in queue.", "task_id": task_id}, status=404)
    
    async def _main_worker(self):
        while True:
            task, args, future, task_id = await self.main_queue.get()

            if task_id not in self.queued_tasks:
                logger.debug(f"Task {task_id} was cancelled, skipping.")
                if future:
                    future.set_exception(asyncio.CancelledError(f"Task {task_id} was cancelled"))
                self.main_queue.task_done()
                continue

            current_task = self.queued_tasks.pop(task_id)

            self.current_task = {
                "task_id": task_id,
                "task": task,
                "name": current_task["name"],
                "sid": current_task["sid"],
                "started_at": time.time(),
                "progress": 0,
                "args": args,
            }
            task_list, current_task = self._get_queue()
            self.queue_message({
                "type": "task_started",
                "task_id": task_id,
                "queued": task_list,
                "current": current_task,
            })
            
            try:
                if isinstance(args, tuple):
                    result = await self.loop.run_in_executor(None, partial(task, *args))
                elif isinstance(args, dict):
                    result = await self.loop.run_in_executor(None, partial(task, **args))
                else:
                    result = await self.loop.run_in_executor(None, partial(task, args))
                
                if future:
                    future.set_result(result)
            except Exception as e:
                #logger.error(f"Error processing main task: {e}")
                logger.error(f"Error occurred in {traceback.format_exc()}")
                if future:
                    future.set_exception(e)
            finally:
                if self.current_task:
                    task_list, _ = self._get_queue()
                    self.queue_message({
                        "type": "task_completed",
                        "task_id": task_id,
                        "completed_at": time.time(),
                        "queued": task_list,
                        "current": None,
                        "sid": self.current_task["sid"],
                        "args": args,
                    })
                    self.current_task = None
                self.main_queue.task_done()
                self.interrupt_flag = False


    async def _background_worker(self):
        while True:
            task, args = await self.background_queue.get()
            try:
                if isinstance(args, tuple):
                    await task(*args)
                elif isinstance(args, dict):
                    await task(**args)
                else:
                    await task(args)
            except Exception as e:
                logger.error(f"Error processing background task: {e}")
                #logger.error(f"Error occurred in {traceback.format_exc()}")
            finally:
                self.background_queue.task_done()


    """
    ╭─────────────────────╮
       Basic HTTP Routes   
    ╰─────────────────────╯
    """

    async def index(self, _):
        response = web.FileResponse('web/index.html')
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def favicon(self, _):
        return web.FileResponse('web/favicon.ico')

    async def user_assets(self, request):
        module = request.match_info.get('module')
        file = request.match_info.get('file')
        fileName = f"custom/{module}/web/{file}"
        
        if not Path(fileName).exists():
            return web.HTTPNotFound(text='File not found')
        
        response = web.FileResponse(fileName)
        #response.headers["Content-Type"] = "application/javascript"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response


    """
    ╭─────────────────────────╮
       Nodes & Fields Routes
    ╰─────────────────────────╯
    """
    async def nodes(self, request):
        id = request.match_info.get('id', '').strip('/')
        modules = self.modules

        if id:
            m, n = id.split('/')
            if m not in modules:
                return web.json_response({"error": f"The module {m} was not found. Try refreshing the page and restarting the server."}, status=404)
            if n not in modules[m]:
                return web.json_response({"error": f"The node {n} was not found in the module {m}. Try refreshing the page and restarting the server."}, status=404)
            modules = {m: {n: modules[m][n]}}

        output = {}
        for module, actions in modules.items():
            for action, values in actions.items():
                params = deepcopy(values.get('params', {}))
                #spawn_fields = []
                for p in params:
                    if 'postProcess' in params[p]:
                        del params[p]["postProcess"]
                    #if 'spawn' in params[p] and params[p]['spawn']:
                    #    spawn_fields.append(p)
                # spawn fields are identified by the '>>>' suffix in the field key
                #for p in spawn_fields:
                #    params[f"{p}>>>0"] = params[p]
                #    del params[p]

                output[f"{module}.{action}"] = {
                    'module': module,
                    'action': action,
                    'type': values.get('type', 'custom'),
                    'label': values.get('label', f"{module}: {action}"),
                    'category': values.get('category', 'default'),
                    'description': values.get('description', ''),
                    'style': values.get('style', ''),
                    'resizable': values.get('resizable', False),
                    'time': [0,0,0],
                    'memory': [0,0,0],
                    'cache': False,
                    'params': params
                }
        
        return web.json_response({
            'instance': self.instance,
            'nodes': output
        })

    async def field_action(self, request):
        data = await request.json()
        node = data.get('node')
        sid = data.get('sid')
        fn = data.get('fn')
        values = data.get('values')
        key = data.get('fieldKey', None)
        queue = data.get('queue', False)

        if node not in self.node_cache:
            module = data.get('module')
            action = data.get('action')
            work_module = import_module(f"{module}.main")
            work_action = getattr(work_module, action)
            work_action = work_action(node_id=node)
            self.node_cache[node] = work_action
        
        self.node_cache[node]._sid = sid # always update the sid as it may change over time

        fn = getattr(self.node_cache[node], fn)
        ref = {
            "node": node,
            "key": key,
            "queue": queue,
        }

        if queue:
            task_id = await self.queue_task(fn, (values, ref), None, sid, name=f"Field action")
        else:
            fn(values, ref)
            task_id = None

        return web.json_response({
            "error": False,
            "message": f"Field action `{fn}` for node `{node}` queued for processing",
            "sid": sid,
            "task_id": task_id,
            "ref": ref,
        })

    """
    ╭─────────────────────╮
       Node Cache Routes
    ╰─────────────────────╯
    """

    async def cache(self, request):
        node = request.match_info.get('node')
        field = request.match_info.get('field')
        index = request.match_info.get('index', None)

        if node not in self.node_cache:
            return web.HTTPNotFound(text=f"Node {node} not found in cache.")

        # get the actual value from the node cache
        if field in self.node_cache[node].output:
            data = self.node_cache[node].output[field]
        elif field in self.node_cache[node].params:
            data = self.node_cache[node].params[field]
        else:
            return web.HTTPNotFound(text=f"Field {field} not found in node {node} cache.")
        
        if data is None:
            return web.HTTPNotFound(text=f"Field {field} is empty in node {node} cache.")
        
        if isinstance(data, list):
            index = max(0, min(len(data) - 1, int(index))) if index else 0
            data = data[index]

        # check the registry for the type of the field
        module = self.node_cache[node].module_name
        action = self.node_cache[node].class_name
        type = self.modules[module][action]['params'][field].get('type')

        filename = request.query.get('filename', f"{field}")

        charset = None
        if type == 'image':
            format = request.query.get('format', 'WEBP').upper()
            quality = request.query.get('quality')
            out = to_bytes(type, data, {'format': format, 'quality': quality})
            content_type = f'image/{format.lower()}'
            filename = f"{filename}.{format.lower()}"
        elif type == 'text' or type.startswith('str'):
            out = str(data).encode('utf-8')
            content_type = f'text/plain'
            charset = 'utf-8'
            filename = f"{filename}.txt"
        else:
            return web.HTTPBadRequest(text=f"Data type `{type}` not supported.")
        
        return web.Response(
            body=out,
            content_type=content_type,
            charset=charset,
            headers={
                'Content-Disposition': f'inline; filename="{filename}"',
                'Content-Length': str(len(out)),
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )

    async def delete_cache(self, request):
        data = await request.json()
        nodes = data.get('nodes', [])

        if isinstance(nodes, str):
            nodes = list(self.node_cache.keys()) if nodes == '*' else [nodes]

        # this might take a while because it could be freeing up VRAM
        for node in nodes:
            if node in self.node_cache:
                del self.node_cache[node]
        
        logger.debug(f"Removed {len(nodes)} nodes from cache.")
        
        return web.json_response({"error": False, "nodes": nodes})
    

    """
    ╭───────────────────╮
       File Management
    ╰───────────────────╯
    """

    async def listdir(self, request):
        req_path = request.query.get('path', self.work_dir)
        req_type = request.query.get('type', None)
        if req_type:
            req_type = [t.strip() for t in req_type.lower().split(',')]

        full_path = Path(req_path)
        if not full_path.is_absolute():
            full_path = Path(self.work_dir) / full_path

        file_types = {
            'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'ico', 'webp'],
            'audio': ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma', 'm4b', 'm4p', 'm4r'],
            'video': ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'mpeg', 'mpg', 'm4v', 'webm'],
            'text': ['txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'log', 'html', 'css', 'js', 'ts', 'py', 'rb', 'php', 'sql', 'sh', 'bash'],
            'archive': ['zip', 'rar', 'tar', 'gz', 'bz2', '7z'],
            '3d': ['glb', 'gltf', 'stl', 'obj', 'fbx', 'dae', 'ply', '3ds', 'max', 'blend'],
        }

        if not str(full_path).startswith(self.work_dir):
            return web.json_response({"error": f"Cannot access paths outside of {self.work_dir}."}, status=403)

        contents = {
            'files': [],
            'path': '',
            'abs_path': '',
        }

        try:
            if full_path.exists():
                contents['path'] = str(full_path.relative_to(self.work_dir))
                contents['abs_path'] = str(full_path)

                for item in full_path.iterdir():
                    suffix = item.suffix.lstrip('.').lower()
                    # if any of the requested types don't match the file type, skip it
                    if not item.is_dir() and req_type and not any(suffix in exts for ftype, exts in file_types.items() if ftype in req_type):
                        continue

                    file = {
                        'is_dir': item.is_dir(),
                        'is_hidden': item.name.startswith('.'), # TODO: Windows: bool(os.stat(item).st_mode & stat.FILE_ATTRIBUTE_HIDDEN),
                        'name': item.name,
                        'path': str(item.relative_to(self.work_dir)),
                        #'abs_path': str(item),
                        'modified': item.stat().st_mtime,
                        'size': None,
                        'ext': None,
                        'type': None,
                    }
                    if not item.is_dir():
                        file['size'] = item.stat().st_size
                        file['ext'] = suffix
                        file['type'] = next((ftype for ftype, exts in file_types.items() if suffix in exts), 'other')
                    
                    contents['files'].append(file)

                return web.json_response(contents)
            else:
                return web.json_response({"error": f"The path {req_path} does not exist."}, status=404)

        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def filePost(self, request):
        data = await request.post()
        file = data.get('file')
        type = data.get('type', 'images')
        type = type if type in ['images', 'audio', 'video', 'text', '3d'] else 'images'
        file_path = Path(self.data_dir) / type / file.filename

        if file_path.exists():
            file_path = file_path.with_name(f"{file_path.stem}_{nanoid.generate(size=6)}{file_path.suffix}")
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(file.file.read())
            
            return web.json_response({"error": False, "path": str(file_path.relative_to(self.work_dir))})
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def preview(self, request):
        from utils.image import cover
        from PIL import Image
        from io import BytesIO

        Image.MAX_IMAGE_PIXELS = None

        file = request.query.get('file')
        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)
        
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = Path(self.work_dir) / file_path

        if not str(file_path).startswith(self.work_dir):
            return web.json_response({"error": f"Cannot access paths outside of {self.work_dir}."}, status=403)

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)
        
        if not file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.webp']:
            return web.json_response({"error": f"The file {file} is not an image."}, status=400)

        width = int(request.query.get('width', 0))
        height = int(request.query.get('height', 0))
        format = request.query.get('format', 'jpeg')
        format = format.lower()
        quality = int(request.query.get('quality', 95))

        image = Image.open(file_path)

        if width > 0 or height > 0:
            width = min(2048, width) if width > 0 else min(2048, height)
            height = min(2048, height) if height > 0 else min(2048, width)
        else:
            width = min(2048, image.width)
            height = min(2048, image.height)

        if width != image.width or height != image.height:
            image = cover(image, width, height, resample='BICUBIC')
        
        if image.mode == 'RGBA' and format in ['jpeg', 'jpg', 'bmp', 'ico']:
            image = image.convert('RGB')

        bytes = BytesIO()
        image.save(bytes, format=format.upper(), quality=quality)
        bytes = bytes.getvalue()
        return web.Response(
            body=bytes,
            content_type=f'image/{format.lower()}',
            headers={
                'Content-Disposition': f'inline; filename="{file_path.name}"',
                'Content-Length': str(len(bytes)),
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )

    """
    ╭────────────────────────╮
       Main graph execution
    ╰────────────────────────╯
    """

    async def graph(self, request):
        graph = await request.json()
        sid = graph.get("sid")
        #if not sid:
        #    return web.json_response({"error": True, "message": "Missing session id"}, status=400)
        
        task_id = await self.queue_task(self.execute_graph, (graph,), None, sid, name=f"Graph execution")
        return web.json_response({
            "error": False,
            "message": "Graph queued for processing",
            "sid": sid,
            "task_id": task_id,
        })
    
    def execute_graph(self, graph):
        sid = graph['sid']
        nodes = graph['nodes']
        paths = graph['paths']

        graph_execution_time = time.time()

        node_count = sum(len(path) for path in paths)
        task_progress_step = 100 / node_count if node_count > 0 else 100
        task_progress = 0

        for path in paths:
            for id in path:
                if self.interrupt_flag:
                    self.interrupt_flag = False
                    self.queue_message({
                        "type": "graph_stopped",
                        "sid": sid,
                        "message": "Execution interrupted by the user."
                    }, sid)
                    return

                self.execute_node(id, nodes[id], sid)

                # broadcast the task progress
                if self.current_task:
                    task_progress += task_progress_step
                    self.current_task['progress'] = int(task_progress)
                    self.queue_message({
                        "type": "task_progress",
                        "task_id": self.current_task["task_id"],
                        "progress": self.current_task['progress'],
                    })

        # the graph has completed
        self.queue_message({
            "type": "graph_completed",
            "sid": sid,
            "executionTime": time.time() - graph_execution_time,
        }, sid)

    async def stop_execution(self, _):
        # check if there is a current task or any queued task
        if not self.current_task and not self.queued_tasks:
            return web.json_response({"error": True, "message": "Nothing to do. No task is currently running or queued."})

        if self.interrupt_flag:
            return web.json_response({"error": True, "message": "Execution is already set for interruption."})

        self.interrupt_flag = True

        # set the interrupt flag for all the nodes in the cache
        for node in self.node_cache:
            self.node_cache[node]._interrupt = True
        
        return web.json_response({"error": False, "message": "Execution set for interruption."})

    def execute_node(self, id, node, sid):
        module = node['module']
        action = node['action']
        params = node['params']

        if module not in self.modules:
            raise ValueError(f"Invalid module: {module}")

        if action not in self.modules[module]:
            raise ValueError(f"Invalid action: {action}")

        # get the arguments values
        args = {}
        ui_fields = {}

        for p in params:
            data_source_id = params[p].get('sourceId')
            data_param_key = params[p].get('sourceKey')

            # the field is a UI element, used mostly to display the data in the UI
            if 'display' in params[p] and params[p]['display'] in ['ui_text', 'ui_image', 'ui_audio', 'ui_video', 'ui_3d', 'ui_label', 'ui_button'] and data_param_key:
                ui_fields[p] = data_param_key
            # the field is an input that gets its value from an output of another node
            elif data_source_id and data_param_key:
                # spawn field handling
                #if '>>>' in p or self.modules[module][action]['params'][p].get('spawn'):
                if params[p].get('spawn'):
                    spawn_key = p.split('>>>')[0]
                    if not spawn_key in args:
                        args[spawn_key] = []

                    args[spawn_key].append(self.node_cache[data_source_id].output[data_param_key])
                else:
                    args[p] = self.node_cache[data_source_id].output[data_param_key]
            # the field is a static value
            else:
                args[p] = params[p].get('value')
                    
        reset_memory_stats()
        
        start_time = time.time()

        # tell the client that the node is running
        self.queue_message({
            "type": "progress",
            "node": id,
            "progress": -1, # -1 sets the progress to indeterminate
        }, sid)

        # import the custom module
        work_module = import_module(f"{module}.main")
        work_action = getattr(work_module, action)

        # if the node is not in the cache, initialize it
        if id not in self.node_cache:
            self.node_cache[id] = work_action(id)

        if not callable(self.node_cache[id]):
            raise TypeError(f"The class `{module}.{action}` is not callable. Make sure the class has a `__call__` method or extends `NodeBase`.")

        # set the session id, it can be used to send messages from the node back to the client
        self.node_cache[id]._sid = sid

        # *** execute the node ***
        self.node_cache[id](**args)

        execution_time = time.time() - start_time
        self.node_cache[id]._execution_time['last'] = execution_time
        self.node_cache[id]._execution_time['min'] = min(self.node_cache[id]._execution_time['min'], execution_time) if self.node_cache[id]._execution_time['min'] is not None else execution_time
        self.node_cache[id]._execution_time['max'] = max(self.node_cache[id]._execution_time['max'], execution_time) if self.node_cache[id]._execution_time['max'] is not None else execution_time

        memory_stats = get_memory_stats()
        if memory_stats:
            self.node_cache[id]._memory_usage['last'] = memory_stats['peak']
            self.node_cache[id]._memory_usage['min'] = min(self.node_cache[id]._memory_usage['min'], memory_stats['peak']) if self.node_cache[id]._memory_usage['min'] is not None else memory_stats['peak']
            self.node_cache[id]._memory_usage['max'] = max(self.node_cache[id]._memory_usage['max'], memory_stats['peak']) if self.node_cache[id]._memory_usage['max'] is not None else memory_stats['peak']

        # the node has completed
        self.queue_message({
            "type": "executed",
            "node": id,
            "name": f"{module}.{action}",
            "hasChanged": self.node_cache[id]._has_changed,
            "executionTime": self.node_cache[id]._execution_time,
            "memoryUsage": self.node_cache[id]._memory_usage
        }, sid)

        for ui_key, data_key in ui_fields.items():
            message = None

            # skip for button fields
            if self.modules[module][action]['params'][ui_key].get('display') == 'ui_button':
                continue

            # if the data key is an output, get the value from the output otherwise from the params
            if data_key in self.node_cache[id].output:
                source_value = self.node_cache[id].output[data_key]
            else:
                source_value = self.node_cache[id].params[data_key]
                        
            data_type = self.modules[module][action]['params'][data_key].get('type') # data type of the source field
            data_format = self.modules[module][action]['params'][ui_key].get('type', 'text') # format of the returned value: text, raw, url
            fieldOptions = self.modules[module][action]['params'][ui_key].get('fieldOptions', {})

            source_value = source_value if isinstance(source_value, list) else [source_value]
            if data_format == 'url':
                if data_type == 'image':
                    data_value = [f"/cache/{id}/{data_key}/{i}?format={fieldOptions.get('format', 'WEBP')}&quality={fieldOptions.get('quality', 100)}&t={time.time()}"
                                  for i in range(len(source_value)) if source_value[i] is not None]
                else:
                    data_value = [f"/cache/{id}/{data_key}/{i}?t={time.time()}"
                                  for i in range(len(source_value)) if source_value[i] is not None]
            elif data_format == 'raw':
                data_value = [to_bytes(data_type, item, fieldOptions) for item in source_value if item is not None]
            else:
                data_value  = [to_base64(data_type, item, fieldOptions) for item in source_value if item is not None]

            message = {
                'client_id': sid,
                'type': 'update_value',
                'node': id,
                'key': ui_key,
                'data_type': data_type,
                'value': data_value
            }

            if message:
                self.queue_message(message, sid)


    def trigger_node(self, source_id, output, sid):
        if not self.current_task:
            return

        graph = self.current_task['args'][0]
        nodes = graph['nodes']

        for id in nodes:
            params = nodes[id]['params']

            for p in params:
                data_source_id = params[p].get('sourceId')
                data_param_key = params[p].get('sourceKey')
                if data_source_id == source_id and data_param_key == output:
                    self.execute_node(id, nodes[id], sid)

    """
    ╭────────────────╮
       Hugging Face
    ╰────────────────╯
    """

    async def hf_cache(self, request):
        return web.json_response(get_local_models())

    async def hf_cache_delete(self, request):
        hashes = request.match_info.get('hash').split(',')
        if not hashes:
            return web.json_response({"error": "Incorrect request, `hash` is required."}, status=400)
        
        result = delete_model(*hashes)
        return web.json_response({"error": not result})

    # TODO: not yet implemented
    async def hf_hub(self, request):
        query = request.query.get('q', '')

        future = asyncio.Future()
        await self.main_queue.put((search_hub, query, future))
        
        try:
            result = await future
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error in hf_hub endpoint: {e}")
            return web.json_response({"error": str(e)}, status=500)

    # TODO: not yet implemented
    async def hf_download(self, request):
        repo_id = request.query.get('repo_id')

        if not repo_id:
            return web.json_response({"error": "Incorrect request, `repo_id` is required."}, status=400)
        
        def progress_cb(progress):
            print(f"Download progress: {progress * 100:.1f}%")
        
        future = asyncio.Future()
        await self.main_queue.put((download_hub_model, (repo_id, progress_cb), future))
        
        try:
            result = await future
            return web.json_response({"error": False, "result": result})
        except Exception as e:
            logger.error(f"Error in hf_download endpoint: {e}")
            return web.json_response({"error": str(e)}, status=500)


    """
    ╭───────────────╮
        Websocket    
    ╰───────────────╯
    """    
    async def websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = request.query.get('sid')
        if not sid:
            sid = nanoid.generate(size=10)
        if sid in self.ws_sessions:
            del self.ws_sessions[sid]
        
        self.ws_sessions[sid] = ws
        logger.debug(f"Websocket connection opened: {sid}")

        # send the welcome message together with the ids of the cached nodes
        await self.broadcast({"type": "welcome", "instance": self.instance, "sid": sid, "cachedNodes": list(self.node_cache.keys())}, sid)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data['type'] == 'close':
                        await ws.close()
                        break
                    elif data['type'] == 'ping':
                        await self.broadcast({"type": "pong"}, sid)
                    else:
                        logger.error(f"[Websocket] Invalid message type: {data['type']}")
                elif msg.type == WSMsgType.CLOSE:
                    await self.broadcast({"type": "close"}, sid)
                    break
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"[Websocket] Error: {ws.exception()}")
        except Exception as e:
            logger.error(f"[Websocket] Error: {e}")
        finally:
            if sid in self.ws_sessions:
                del self.ws_sessions[sid]
            logger.debug(f"Websocket connection closed: {sid}")
        
        return ws

    async def broadcast(self, message: dict | bytes, sid: list[str] | str = None, exclude: list[str] | str = None):
        sessions = []

        if sid:
            sessions = [sid] if not isinstance(sid, list) else sid
        else:
            sessions = list(self.ws_sessions.keys())

        if exclude:
            exclude = [exclude] if not isinstance(exclude, list) else exclude
            sessions = [s for s in sessions if s not in exclude]

        for session in sessions:
            try:
                if session in self.ws_sessions:
                    if isinstance(message, dict):
                        await self.ws_sessions[session].send_json(message)
                    else:
                        await self.ws_sessions[session].send_bytes(message)
            except Exception as e:
                logger.error(f"[Websocket] Error broadcasting message: {e}")
                pass
    
    def queue_message(self, message: dict | bytes, sid: list[str] | str = None, exclude: list[str] | str = None):
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.background_queue.put((self.broadcast, (message, sid, exclude))),
                self.loop
            )


def to_base64(type, value, options={}):
    import io
    import base64
    
    out = value

    if type == 'image':
        format = options.get('format', 'WEBP').upper()
        quality = options.get('quality')
        if format == 'WEBP' and not quality:
            quality = 100
        elif format == 'JPEG' and not quality:
            quality = 75
        elif format == 'PNG' and not quality:
            quality = None
        mime_type = f"image/{format.lower()}"

        byte_arr = io.BytesIO()
        value.save(byte_arr, format=format, quality=int(quality))
        # TODO: check shutil.copyfile
        header = f"data:{mime_type};base64,"
        out = header + base64.b64encode(byte_arr.getvalue()).decode('utf-8')
    
    return out

def to_bytes(data_type, value, options={}):
    import io
    from PIL import Image
    
    out = value

    if isinstance(value, Image.Image):
        format = options.get('format', 'WEBP').upper()
        quality = options.get('quality')
        if format == 'WEBP' and not quality:
            quality = 100
        elif format == 'JPEG' and not quality:
            quality = 75
        elif format == 'PNG' and not quality:
            quality = None

        byte_arr = io.BytesIO()
        value.save(byte_arr, format=format, quality=int(quality))
        out = byte_arr.getvalue()
    elif isinstance(value, str):
        out = value.encode('utf-8')
    
    return out

server = WebServer(
    MODULE_MAP,
    host=CONFIG.server['host'],
    port=CONFIG.server['port'],
    secure=CONFIG.server['secure'],
    certfile=CONFIG.server['certfile'],
    keyfile=CONFIG.server['keyfile'],
    cors=CONFIG.server['cors'],
    cors_routes=CONFIG.server['cors_routes'],
    client_max_size=CONFIG.server['client_max_size'],
    work_dir=CONFIG.paths['work_dir'],
    data_dir=CONFIG.paths['data']
)
