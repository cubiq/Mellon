import logging
import asyncio
from functools import partial
from importlib import import_module
logger = logging.getLogger('mellon')
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

        self.node_cache = {}
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
            web.get('/cache/{node}/{field}', self.cache),
            web.get('/cache/{node}/{field}/{index}', self.cache),
            web.delete('/cache', self.delete_cache),
            web.get('/listdir', self.listdir),
            web.post('/file', self.filePost),
            web.get('/preview', self.preview),
            web.post('/graph', self.graph),
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
        # Close all websocket sessions
        for ws in list(self.ws_sessions.values()):
            try:
                await ws.close()
            except Exception:
                pass
        self.ws_sessions.clear()

        # Cancel worker tasks
        if self.main_worker_task:
            self.main_worker_task.cancel()
        if self.background_worker_task:
            self.background_worker_task.cancel()

        # Wait for the workers to finish
        tasks_to_wait = [task for task in [self.main_worker_task, self.background_worker_task] if task]
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

        # Clean up the runner and site
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    """
    ╭───────────────╮
          Queue    
    ╰───────────────╯
    """
        
    async def _main_worker(self):
        while True:
            task, args, future = await self.main_queue.get()
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
                logger.error(f"Error processing main task: {e}")
                logger.error(f"Error occurred in {traceback.format_exc()}")
                if future:
                    future.set_exception(e)
            finally:
                self.main_queue.task_done()

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
    ╭───────────────╮
       HTTP Routes   
    ╰───────────────╯
    """

    async def index(self, _):
        response = web.FileResponse('web/index.html')
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def favicon(self, _):
        return web.FileResponse('web/favicon.ico')

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
                for p in params:
                    if 'postProcess' in params[p]:
                        del params[p]["postProcess"]

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
        
        if isinstance(data, list):
            index = max(0, min(len(data) - 1, int(index))) if index else 0
            data = data[index]

        # check the registry for the type of the field
        module = self.node_cache[node].module_name
        action = self.node_cache[node].class_name
        type = self.modules[module][action]['params'][field].get('type')

        filename = request.rel_url.query.get('filename', f"{field}")

        charset = None
        if type == 'image':
            out = to_bytes(type, data)
            content_type = f'image/webp'
            filename = f"{filename}.webp"
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

        for node in nodes:
            if node in self.node_cache:
                del self.node_cache[node]
        
        return web.json_response({"error": False, "nodes": nodes})

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

    # Execute the node graph
    async def graph(self, request):
        graph = await request.json()
        await self.main_queue.put((self.execute_graph, (graph,), None))
        return web.json_response({
            "error": False,
            "message": "Graph queued for processing",
            "sid": graph["sid"]
        })
    
    def execute_graph(self, graph):
        sid = graph['sid']
        nodes = graph['nodes']
        paths = graph['paths']

        graph_execution_time = time.time()

        for path in paths:
            for id in path:
                module = nodes[id]['module']
                action = nodes[id]['action']
                params = nodes[id]['params']

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
                    if 'display' in params[p] and params[p]['display'] in ['ui_text', 'ui_image', 'ui_audio', 'ui_video', 'ui_3d'] and data_param_key:
                        ui_fields[p] = data_param_key
                    # the field is an input that gets its value from an output of another node
                    elif data_source_id and data_param_key:
                        # spawn field handling
                        if '>>>' in p or self.modules[module][action]['params'][p].get('spawn'):
                            spawn_key = p.split('>>>')[0]
                            if not spawn_key in args:
                                args[spawn_key] = []
    
                            args[spawn_key].append(self.node_cache[data_source_id].output[data_param_key])
                        else:
                            args[p] = self.node_cache[data_source_id].output[data_param_key]
                    # the field is a static value
                    else:
                        args[p] = params[p].get('value')
                            
                # import the custom module
                work_module = import_module(f"{module}.main")
                work_action = getattr(work_module, action)

                # tell the client that the node is running
                self.queue_message({
                    "type": "progress",
                    "node": id,
                    "progress": -1, # -1 sets the progress to indeterminate
                }, sid)

                start_time = time.time()

                # if the node is not in the cache, initialize it 
                if id not in self.node_cache:
                    self.node_cache[id] = work_action(id)

                if not callable(self.node_cache[id]):
                    raise TypeError(f"The class `{module}.{action}` is not callable. Make sure the class has a `__call__` method or extends `NodeBase`.")

                # set the session id, it can be used to send messages directly from the node back to the client
                self.node_cache[id]._sid = sid

                # execute the node
                self.node_cache[id](**args)

                execution_time = time.time() - start_time
                self.node_cache[id]._execution_time['last'] = execution_time
                self.node_cache[id]._execution_time['min'] = min(self.node_cache[id]._execution_time['min'], execution_time) if self.node_cache[id]._execution_time['min'] is not None else execution_time
                self.node_cache[id]._execution_time['max'] = max(self.node_cache[id]._execution_time['max'], execution_time) if self.node_cache[id]._execution_time['max'] is not None else execution_time

                # the node has completed
                self.queue_message({
                    "type": "executed",
                    "node": id,
                    "executionTime": self.node_cache[id]._execution_time
                }, sid)

                for ui_key, data_key in ui_fields.items():
                    # if the data key is an output, get the value from the output otherwise from the params
                    if data_key in self.node_cache[id].output:
                        source_value = self.node_cache[id].output[data_key]
                    else:
                        source_value = self.node_cache[id].params[data_key]
                    data_type = self.modules[module][action]['params'][data_key].get('type') # data type of the source field
                    data_format = self.modules[module][action]['params'][ui_key].get('type', 'json') # json, raw, url
                    message = None

                    if data_format == 'json' or data_format == 'url':
                        #data_value = to_base64(data_type, source_value) if data_format == 'json' else f"/cache/{id}/{data_key}?t={time.time()}"
                        source_value = source_value if isinstance(source_value, list) else [source_value]
                        if data_format == 'json':
                            data_value  = [to_base64(data_type, item) for item in source_value]
                        else:
                            data_value = [f"/cache/{id}/{data_key}/{i}?t={time.time()}" for i in range(len(source_value))]

                        message = {
                            'client_id': sid,
                            'type': 'update_value',
                            'node': id,
                            'key': ui_key,
                            'data_type': data_type,
                            'value': data_value
                        }
                    elif data_format == 'raw':
                        # TODO: add support for raw data
                        message = to_bytes(data_type, source_value)

                    if message:
                        self.queue_message(message, sid)

        # the graph has completed
        self.queue_message({
            "type": "graph_completed",
            "sid": sid,
            "executionTime": time.time() - graph_execution_time,
        }, sid)

    async def hf_cache(self, request):
        return web.json_response(get_local_models())

    async def hf_cache_delete(self, request):
        hashes = request.match_info.get('hash').split(',')
        if not hashes:
            return web.json_response({"error": "Incorrect request, `hash` is required."}, status=400)
        
        result = delete_model(*hashes)
        return web.json_response({"error": not result})

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
        asyncio.run_coroutine_threadsafe(
            self.background_queue.put((self.broadcast, (message, sid, exclude))),
            self.loop
        )


def to_base64(type, value):
    import io
    import base64
    
    out = value

    if type == 'image':
        byte_arr = io.BytesIO()
        value.save(byte_arr, format='WEBP', quality=100)
        header = f"data:image/webp;base64,"
        out = header + base64.b64encode(byte_arr.getvalue()).decode('utf-8')
    
    return out

def to_bytes(type, value):
    import io
    
    out = value

    if type == 'image':
        byte_arr = io.BytesIO()
        value.save(byte_arr, format='WEBP', quality=100)
        out = byte_arr.getvalue()
    elif isinstance(value, bytes):
        out = value
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
