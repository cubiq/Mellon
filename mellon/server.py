import logging
logger = logging.getLogger('mellon')
from aiohttp import web, WSMsgType
from aiohttp_cors import setup as cors_setup, ResourceOptions
import json
import nanoid
import io
import base64
import re
from importlib import import_module
import asyncio
import traceback
from utils.memory_manager import memory_flush
from copy import deepcopy
import random
import signal
import time
import sys

class WebServer:
    def __init__(self, module_map: dict, host: str = "0.0.0.0", port: int = 8080, cors: bool = False, cors_route: str = "*"):
        self.module_map = module_map
        self.node_store = {}
        self.queue = asyncio.Queue()
        self.queue_task = None
        self.host = host
        self.port = port
        self.ws_clients = {}
        self.app = web.Application()
        self.event_loop = None

        self.client_queue = asyncio.Queue()
        self.client_task = None

        self.app.add_routes([web.get('/', self.index),
                             web.get('/nodes', self.nodes),
                             web.get('/view/{format}/{node}/{key}/{index}', self.view),
                             web.get('/custom_component/{module}/{component}', self.custom_component),
                             web.get('/custom_assets/{module}/{file_path}', self.custom_assets),
                             web.post('/graph', self.graph),
                             web.post('/nodeExecute', self.node_execute),
                             web.delete('/clearNodeCache', self.clear_node_cache),
                             web.static('/assets', 'web/assets'),
                             web.get('/favicon.ico', self.favicon),
                             web.get('/ws', self.websocket_handler)])

        if cors:
            cors = cors_setup(self.app, defaults={
                cors_route: ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            })

            for route in list(self.app.router.routes()):
                cors.add(route)

    def run(self):
        async def shutdown():
            if hasattr(self, 'is_shutting_down') and self.is_shutting_down:
                return
            self.is_shutting_down = True
            
            logger.info("Received shutdown signal. Namárië!")
            self.shutdown_event.set()

            # Cancel all running tasks except the current one
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()

            #try:
                # Wait for all tasks to finish
            await asyncio.gather(*tasks, return_exceptions=True)
            #except asyncio.CancelledError:
                #pass  # Ignore cancelled error during shutdown

            # Close all websocket connections
            for ws in list(self.ws_clients.values()):
                try:
                    await ws.close(code=1000, message=b'Server shutting down')
                except Exception:
                    pass  # Ignore any websocket closing errors
            self.ws_clients.clear()

        async def start_app():
            self.shutdown_event = asyncio.Event()
            self.event_loop = asyncio.get_event_loop()
            self.is_shutting_down = False
            
            if hasattr(signal, "SIGBREAK"):  # Windows
                def signal_handler(signum, frame):
                    self.event_loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(shutdown())
                    )
                
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGBREAK):
                    signal.signal(sig, signal_handler)
            else:  # Unix/Linux/MacOS
                for sig in (signal.SIGINT, signal.SIGTERM):
                    self.event_loop.add_signal_handler(
                        sig,
                        lambda: asyncio.create_task(shutdown())
                    )

            runner = web.AppRunner(self.app, client_max_size=1024**4)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)

            # Start background tasks
            self.queue_task = asyncio.create_task(self.process_queue())
            self.client_task = asyncio.create_task(self.process_client_messages())

            await site.start()

            try:
                await self.shutdown_event.wait()
            finally:
                await shutdown()
                await runner.cleanup()

        try:
            asyncio.run(start_app())
        except KeyboardInterrupt:
            # On Windows, asyncio.run() may not handle KeyboardInterrupt properly
            pass
        except asyncio.CancelledError:
            pass  # Ignore cancelled error during shutdown

    async def process_client_messages(self):
        while True:
            message = await self.client_queue.get()
            try:
                await self.ws_clients[message["client_id"]].send_json(message["data"])
            except Exception as e:
                logger.error(f"Error sending client message: {str(e)}")
            finally:
                self.client_queue.task_done()

    async def process_queue(self):
        while True:
            item = await self.queue.get()
            try:
                if "kwargs" in item:
                    await self.node_execute_single(item)
                else:
                    await self.graph_execution(item)
            except Exception as e:
                logger.error(f"Error processing queue task: {str(e)}")
                logger.error(f"Error occurred in {traceback.format_exc()}")
                await self.broadcast({
                    "type": "error",
                    "error": "An unexpected error occurred while processing the graph"
                })
            finally:
                self.queue.task_done()

    async def index(self, request):
        response = web.FileResponse('web/index.html')
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def favicon(self, request):
        return web.FileResponse('web/favicon.ico')

    """
    HTTP API
    """

    async def custom_component(self, request):
        module = request.match_info.get('module')
        component = request.match_info.get('component')

        path = component.split('/')
        if len(path) > 1:
            module = path[0]
            component = path[1]

        #if module not in self.module_map:
        #    raise web.HTTPNotFound(text=f"Module {module} not found")

        response = web.FileResponse(f'custom/{module}/web/{component}.js')
        response.headers["Content-Type"] = "application/javascript"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def custom_assets(self, request):
        module = request.match_info.get('module')
        file_path = request.match_info.get('file_path')

        #if module not in self.module_map:
        #    raise web.HTTPNotFound(text=f"Module {module} not found")

        return web.FileResponse(f'custom/{module}/web/assets/{file_path}')

    async def nodes(self, request):
        nodes = {}
        for module_name, actions in self.module_map.items():
            for action_name, action in actions.items():
                params = {}
                groups = {}
                if 'params' in action:
                    params = deepcopy(action['params'])

                    for p in params:
                        # remove attributes that are not needed by the client
                        if 'postProcess' in params[p]:
                            del params[p]['postProcess']

                nodes[f"{module_name}-{action_name}"] = {
                    "label": action.get('label', f"{module_name}: {action_name}"),
                    "module": module_name,
                    "action": action_name,
                    "category": self.slugify(action.get('category', 'default')),
                    "params": params,
                    "groups": groups,
                }
                if 'style' in action:
                    nodes[f"{module_name}-{action_name}"]["style"] = action['style']
                if 'resizable' in action:
                    nodes[f"{module_name}-{action_name}"]["resizable"] = action['resizable']

        return web.json_response(nodes)
    
    async def view(self, request):
        allowed_formats = ['webp', 'png', 'jpeg', 'glb', 'text']

        format = request.match_info.get('format', 'webp').lower()
        if format not in allowed_formats:
            raise web.HTTPNotFound(text=f"Invalid format: {format}")

        nodeId = request.match_info.get('node')
        key = request.match_info.get('key')
        node = self.node_store.get(nodeId)

        if node is None:
            raise web.HTTPNotFound(text=f"Node {nodeId} not found")
        if key not in node.output:
            raise web.HTTPNotFound(text=f"Key {key} not found in node {nodeId}")   
        
        value = node.output[key]
        if value is None:
            raise web.HTTPNotFound(text=f"No data found for {key}")
        
        if not isinstance(value, list):
            value = [value]

        index = int(request.match_info.get("index", 0))
        if index < 0 or index >= len(value):
            raise web.HTTPNotFound(text=f"Index {index} out of bounds for {key}")
        
        # get additional request parameters
        quality = int(request.rel_url.query.get("quality", 100))
        quality = max(0, min(100, quality))
        scale = float(request.rel_url.query.get("scale", 1))
        scale = max(0.01, min(2, scale))
        filename = request.rel_url.query.get("filename", f"{key}_{index}.{format}")

        value = value[index]
        if scale != 1:
            from PIL.Image import Resampling
            width = int(value.width * scale)
            height = int(value.height * scale)
            value = value.resize((max(width, 1), max(height, 1)), resample=Resampling.BICUBIC)

        # return the value as image
        if format == "webp" or format == "png" or format == "jpeg":
            byte_arr = io.BytesIO()
            value.save(byte_arr, format=format.upper(), quality=quality)
            byte_arr = byte_arr.getvalue()
            return web.Response(
                body=byte_arr,
                content_type="image/webp",
                headers={
                    "Content-Disposition": f"inline; filename={filename}",
                    "Content-Length": str(len(byte_arr)),
                    "Cache-Control": "max-age=31536000, immutable",
                }
            )
        elif format == "glb":
            byte_arr = io.BytesIO()
            byte_arr.write(value)
            byte_arr = byte_arr.getvalue()
            return web.Response(
                body=byte_arr,
                content_type="model/glb",
                headers={
                    "Content-Disposition": f"inline; filename={key}.glb",
                    "Content-Length": str(len(byte_arr)),
                    "Cache-Control": "max-age=31536000, immutable",
                }
            )
        elif format == "text":
            return web.json_response({ "data": value })

    async def clear_node_cache(self, request):
        data = await request.json()
        nodeId = []

        if "nodeId" in data:
            nodeId = data["nodeId"] if isinstance(data["nodeId"], list) else [data["nodeId"]]
        else:
            nodeId = list(self.node_store.keys())

        for node in nodeId:
            if node in self.node_store:
                self.node_store[node] = None
                del self.node_store[node]

        memory_flush(gc_collect=True)

        return web.json_response({
            "type": "cacheCleared",
            "nodeId": nodeId
        })

    async def graph(self, request):
        graph = await request.json()
        await self.queue.put(graph)
        return web.json_response({
            "type": "graphQueued",
            "sid": graph["sid"]
        })

    async def node_execute(self, request):
        data = await request.json()
        await self.queue.put(data)
        return web.json_response({
            "type": "nodeQueued",
            "sid": data["sid"],
        })

    async def node_execute_single(self, data):
        sid = data["sid"]
        module = data["module"]
        action = data["action"]
        kwargs = data["kwargs"]
        node = data["node"]

        if module not in self.module_map:
            raise ValueError("Invalid module")
        if action not in self.module_map[module]:
            raise ValueError("Invalid action")

        if module.endswith(".custom"):
            module = import_module(f"custom.{module.replace('.custom', '')}.{module.replace('.custom', '')}")
        else:
            module = import_module(f"modules.{module}.{module}")
        action = getattr(module, action)

        if not callable(action):
            raise ValueError("Action is not callable")

        node = action()
        node._client_id = sid

        result = {}

        try:
            result = await self.event_loop.run_in_executor(None, lambda: node(**kwargs))
        except Exception as e:
            logger.error(f"Error executing node {module}.{action}: {str(e)}")
            raise e

        await self.client_queue.put({
            "client_id": sid,
            "data": {
                "type": "single_executed",
                "nodeId": node,
                "module": module,
                "action": action,
                "result": result,
            }
        })

    async def graph_execution(self, graph):
        #graph = await request.json()
        sid = graph["sid"]
        nodes = graph["nodes"]
        paths = graph["paths"]

        randomized_fields = {}
        for path in paths:
            for node in path:
                module_name = nodes[node]["module"]
                action_name = nodes[node]["action"]
                logger.debug(f"Executing node {module_name}.{action_name}")

                params = nodes[node]["params"]
                ui_fields = {}
                args = {}
                for p in params:
                    source_id = params[p].get("sourceId")
                    source_key = params[p].get("sourceKey")

                    if "display" in params[p] and params[p]["display"] == "ui":
                        # store ui fields that need to be sent back to the client
                        if params[p]["type"] == "image" or params[p]["type"] == "3d" or params[p]["type"] == "text":
                            ui_fields[p] = { "source": source_key, "type": params[p]["type"] }
                    else:
                        # handle list values (spawn input fields)
                        # if p ends with [d+], it means that the field is part of a list
                        if source_id and re.match(r".*\[\d+\]$", p):
                            spawn_key = re.sub(r"\[\d+\]$", "", p)
                            if not args.get(spawn_key):
                                args[spawn_key] = []
                            elif not isinstance(args[spawn_key], list):
                                args[spawn_key] = [args[spawn_key]]

                            args[spawn_key].append(self.node_store[source_id].output[source_key])
                        else:
                            # if there is a source id, it means that the value comes from a pipeline,
                            # so we follow the connection to the source node and get the associated value
                            # Otherwise we use the value in the params
                            args[p] = self.node_store[source_id].output[source_key] if source_id else params[p].get("value")

                # check if there is a field with the name __random__<param>
                # randomize the field unless it has been already randomized
                for key in args:
                    if key.startswith('__random__') and args[key] is True:
                        if node not in randomized_fields:
                            randomized_fields[node] = []
                        if key in randomized_fields[node]:
                            continue
                        randomized_fields[node].append(key)

                        random_field = key.split('__random__')[1]
                        args[random_field] = random.randint(0, (1<<53)-1) # TODO: allow min/max values
                        #self.node_store[node].params[random_field] = args[random_field]
                        params[random_field]["value"] = args[random_field]
                        await self.client_queue.put({
                            "client_id": sid,
                            "data": {
                                "type": "updateValues",
                                "nodeId": node,
                                "key": random_field,
                                "value": args[random_field]
                            }
                        })

                if module_name not in self.module_map:
                    raise ValueError("Invalid module")
                if action_name not in self.module_map[module_name]:
                    raise ValueError("Invalid action")

                # import the module and get the action
                if module_name.endswith(".custom"):
                    module = import_module(f"custom.{module_name.replace('.custom', '')}.{module_name.replace('.custom', '')}")
                else:
                    module = import_module(f"modules.{module_name}.{module_name}")
                action = getattr(module, action_name)

                # if the node is not in the node store, initialize it
                if node not in self.node_store:
                    self.node_store[node] = action(node)

                self.node_store[node]._client_id = sid

                if not callable(self.node_store[node]):
                    raise TypeError(f"The class `{module_name}.{action_name}` is not callable. Ensure that the class has a __call__ method or extend it from `NodeBase`.")

                # initialize the progress bar
                await self.client_queue.put({
                    "client_id": sid,
                    "data": {
                        "type": "progress",
                        "nodeId": node,
                        "progress": -1
                    }
                })

                try:
                    await self.event_loop.run_in_executor(None, lambda: self.node_store[node](**args))
                except Exception as e:
                    logger.error(f"Error executing node {module_name}.{action_name}: {str(e)}")
                    raise e

                execution_time = self.node_store[node]._execution_time if hasattr(self.node_store[node], '_execution_time') else 0

                await self.client_queue.put({
                    "client_id": sid,
                    "data": {
                        "type": "executed",
                        "nodeId": node,
                        "time": f"{execution_time:.2f}",
                        #"updateValues": updateValues
                        #"memory": f"{memory_usage/1024**3:.2f}"
                    }
                })

                logger.debug(f"Node {module_name}.{action_name} executed in {execution_time:.3f}s")

                for key in ui_fields:
                    source = ui_fields[key]["source"]
                    source_value = self.node_store[node].output[source]
                    length = len(source_value) if isinstance(source_value, list) else 1
                    format = ui_fields[key]["type"]
                    if format == "image":
                        format = 'webp'
                    elif format == "3d":
                        format = 'glb'
                    else:
                        format = 'text'
                    data = []
                    if format == 'text':
                        data = {
                            "url": f"/view/{format}/{node}/{source}/{0}?t={time.time()}",
                            "value": source_value
                        }
                    else:
                        for i in range(length):
                            if format == 'image':
                                if length > 1:
                                    scale = 0.5 if source_value[i].width > 1024 or source_value[i].height > 1024 else 1
                                else:
                                    scale = 0.5 if source_value[i].width > 2048 or source_value[i].height > 2048 else 1
                                url = f"/view/{format}/{node}/{source}/{i}?scale={scale}&t={time.time()}"
                                data.append({
                                    "url": url,
                                    "width": source_value[i].width,
                                    "height": source_value[i].height
                                })
                            else:
                                url = f"/view/{format}/{node}/{source}/{i}?t={time.time()}"
                                data.append({
                                    "url": url,
                                })

                    await self.client_queue.put({
                        "client_id": sid,
                        "data": {
                            "type": ui_fields[key]["type"],
                            "key": key,
                            "nodeId": node,
                            "data": data
                            #"data": self.to_base64(ui_fields[key]["type"], value)
                        }
                    })

                await asyncio.sleep(0)

        # if updateValues:
        #     await self.client_queue.put({
        #         "client_id": sid,
        #         "data": {
        #             "type": "updateValues",
        #             "values": updateValues
        #         }
        #     })

    """
    WebSocket API
    """

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = request.query.get("sid")
        logger.info(f"WebSocket connection with sid {sid}")
        if sid:
            if sid in self.ws_clients:
                del self.ws_clients[sid]
        else:
            # if the client does not provide a session id, we create one for them one
            sid = nanoid.generate(size=10)

        self.ws_clients[sid] = ws
        await ws.send_json({"type": "welcome", "sid": sid})

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)

                try:
                    if data["type"] == "ping":
                        await ws.send_json({"type": "pong"})
                    elif data["type"] == "close":
                        await ws.close()
                        break
                        """
                    elif data["type"] == "module":
                        module_name = data["module"]
                        action_name = data["action"]
                        params = data["data"] if "data" in data else {}

                        if module_name not in self.module_map or action_name not in self.module_map[module_name]:
                            raise ValueError("Invalid module or action")

                        module = import_module(f"modules.{module_name}.{module_name}")
                        action = getattr(module, action_name)
                        result = await action(**params)
                        await ws.send_json({"type": "result", "result": result})

                    elif data["type"] == "graph":
                        graph = data["graph"]
                        for node in graph["nodes"]:
                            module_name = node["module"]
                            action_name = node["action"]
                            params = node["params"]
                            module = import_module(f"modules.{module_name}.{module_name}")
                            action = getattr(module, action_name)
                            result = await action(**params)
                            await ws.send_json({"type": "result", "result": result})
                        """
                    else:
                        raise ValueError("Invalid message type")

                #except KeyError as e:
                #    await ws.send_json({"type": "error", "message": f"Missing required field: {str(e)}"})
                #except ValueError as e:
                #    await ws.send_json({"type": "error", "message": str(e)})
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    await ws.send_json({"type": "error", "message": "An unexpected error occurred"})

            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket connection closed with exception {ws.exception()}')

        del self.ws_clients[sid]
        logger.info(f'WebSocket connection {sid} closed')

        return ws

    async def broadcast(self, message, client_id=None):
        if client_id:
            if client_id not in self.ws_clients:
                return
            ws_clients = [client_id] if not isinstance(client_id, list) else client_id
        else:
            ws_clients = self.ws_clients

        for client in ws_clients:
            await self.ws_clients[client].send_json(message)


    """
    Helper functions
    """
    def to_base64(self, type, value):
        if type == "image":
            img_byte_arr = io.BytesIO()
            value.save(img_byte_arr, format='WEBP', quality=100)
            img_byte_arr = img_byte_arr.getvalue()
            return base64.b64encode(img_byte_arr).decode('utf-8')
        elif type == "3d":
            glb_byte_arr = io.BytesIO()
            glb_byte_arr.write(value)
            glb_byte_arr = glb_byte_arr.getvalue()
            return base64.b64encode(glb_byte_arr).decode('utf-8')

    def slugify(self, text):
        return re.sub(r'[^\w\s-]', '', text).strip().replace(' ', '-')

from modules import MODULE_MAP
from config import config

web_server = WebServer(MODULE_MAP, **config.server)
