from mellon.config import CONFIG
import aiohttp
import asyncio
import nanoid
from typing import Dict, Optional, Callable, Any
import json
import logging

logger = logging.getLogger('mellon')

class WebSocketClient:
    """
    WebSocket client for real-time communication with Mellon server.
    """
    
    def __init__(self, address: str, sid: str, message_handlers: Dict[str, Callable] = None):
        self.address = address.replace('http://', 'ws://').replace('https://', 'wss://')
        self.sid = sid
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.is_connected = False
        self.message_handlers = message_handlers or {}
        self._listening_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_reconnect = True
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def connect(self):
        if self.is_connected:
            return
            
        try:
            # Create a new session for WebSocket connection
            self._session = aiohttp.ClientSession()
            
            ws_url = f"{self.address}/ws?sid={self.sid}"
            self.ws = await self._session.ws_connect(ws_url)
            self.is_connected = True
            self._reconnect_delay = 1.0  # Reset reconnect delay on successful connection
            logger.info(f"WebSocket connected to {ws_url}")
            
            # Start listening for messages
            self._listening_task = asyncio.create_task(self._message_listener())
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.is_connected = False
            if self._session:
                await self._session.close()
                self._session = None
            if self._should_reconnect:
                await self._schedule_reconnect()
    
    async def disconnect(self):
        self._should_reconnect = False
        
        if self._listening_task:
            self._listening_task.cancel()
            self._listening_task = None
            
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
            
        if self.ws:
            await self.ws.close()
            self.ws = None
            
        if self._session:
            await self._session.close()
            self._session = None
            
        self.is_connected = False
        logger.info("WebSocket disconnected")
    
    async def send_message(self, message: Dict):
        if not self.is_connected or not self.ws:
            raise ConnectionError("WebSocket is not connected")
            
        try:
            await self.ws.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise
    
    def add_message_handler(self, message_type: str, handler: Callable[[Dict], None]):
        self.message_handlers[message_type] = handler
    
    def remove_message_handler(self, message_type: str):
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
    
    async def _message_listener(self):
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON message received: {e}")
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.ws.exception()}")
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.info("WebSocket connection closed by server")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
        finally:
            self.is_connected = False
            if self._should_reconnect:
                await self._schedule_reconnect()
    
    async def _handle_message(self, data: Dict):
        message_type = data.get('type')
        if not message_type:
            logger.warning("Received message without type")
            return
            
        # Call the appropriate handler
        handler = self.message_handlers.get(message_type)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in message handler for type '{message_type}': {e}")
    
    async def _schedule_reconnect(self):
        if self._reconnect_task:
            self._reconnect_task.cancel()
            
        self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self):
        await asyncio.sleep(self._reconnect_delay)
        
        if not self._should_reconnect:
            return
            
        logger.info(f"Attempting to reconnect to WebSocket (delay: {self._reconnect_delay}s)")
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            # Exponential backoff
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            if self._should_reconnect:
                await self._schedule_reconnect()

class Client:    
    def __init__(self, address: str = '', timeout: int = 30, sid: str = None):
        self.address = address or CONFIG.server['scheme'] + '://' + CONFIG.server['host'] + ':' + str(CONFIG.server['port'])
        self.address = self.address.rstrip('/')
        self.sid = sid or nanoid.generate(size=16)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_client: Optional[WebSocketClient] = None

    async def __aenter__(self):
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            # connector = aiohttp.TCPConnector(
            #     limit=100,  # Total connection pool size
            #     limit_per_host=30,  # Connections per host
            #     keepalive_timeout=30,  # Keep connections alive for 30s
            # )
            self._session = aiohttp.ClientSession(
                # connector=connector,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'Mellon-Client/1.0',
                    #'Accept': 'application/json',
                    #'Content-Type': 'application/json',
                    'X-Session-ID': self.sid
                }
            )
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            
        if self._ws_client:
            await self._ws_client.disconnect()
            self._ws_client = None
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        await self._ensure_session()
        
        url = f"{self.address}/{endpoint.lstrip('/')}"
        
        try:
            async with self._session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                
                # Handle different content types
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    return await response.json()
                else:
                    # For non-JSON responses, return text
                    text = await response.text()
                    return {'data': text, 'content_type': content_type}
                    
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP {e.status} error for {method} {url}: {e.message}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Network error for {method} {url}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {method} {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}")
            raise
    
    async def list_nodes(self) -> Dict:
        return await self._make_request('GET', '/nodes')
    
    async def queue_graph(self, graph: Dict | str) -> Dict:
        if isinstance(graph, str):
            try:
                with open(graph, "r") as f:
                    graph = json.load(f)
            except Exception:
                logger.error(f"Error loading graph from file {graph}")
                raise
        graph['sid'] = self.sid
            
        return await self._make_request('POST', '/graph', json=graph)

    """
    ╭───────────────╮
        Websocket    
    ╰───────────────╯
    """  
    async def connect_websocket(self, message_handlers: Dict[str, Callable] = None) -> WebSocketClient:
        if self._ws_client and self._ws_client.is_connected:
            return self._ws_client
            
        self._ws_client = WebSocketClient(self.address, self.sid, message_handlers)
        await self._ws_client.connect()
        return self._ws_client
    
    async def disconnect_websocket(self):
        if self._ws_client:
            await self._ws_client.disconnect()
            self._ws_client = None
    
    def add_websocket_handler(self, message_type: str, handler: Callable[[Dict], None]):
        if self._ws_client:
            self._ws_client.add_message_handler(message_type, handler)
        else:
            logger.warning("WebSocket client not connected. Call connect_websocket() first.")
    
    def remove_websocket_handler(self, message_type: str):
        if self._ws_client:
            self._ws_client.remove_message_handler(message_type)
        else:
            logger.warning("WebSocket client not connected. Call connect_websocket() first.")
    
    async def send_websocket_message(self, message: Dict):
        if self._ws_client and self._ws_client.is_connected:
            await self._ws_client.send_message(message)
        else:
            raise ConnectionError("WebSocket is not connected. Call connect_websocket() first.")


class ClientSync:
    """
    Synchronous wrapper around the async Mellon client.
    """
    
    def __init__(self, address: str = '', timeout: int = 30, sid: str = None):
        self.address = address or CONFIG.server['scheme'] + '://' + CONFIG.server['host'] + ':' + str(CONFIG.server['port'])
        self.address = self.address.rstrip('/')
        self.timeout = timeout
        self.sid = sid or nanoid.generate(size=16)
    
    def _run_async(self, coro):
        return asyncio.run(coro)
    
    def _sync_wrapper(self, async_func: Callable, *args, **kwargs) -> Any:
        async def _awaitable():
            async with Client(self.address, self.timeout, self.sid) as client:
                return await async_func(client, *args, **kwargs)
        
        return self._run_async(_awaitable())

    def list_nodes(self) -> Dict:
        return self._sync_wrapper(Client.list_nodes)
    
    def queue_graph(self, graph: Dict) -> Dict:
        return self._sync_wrapper(Client.queue_graph, graph)

class Graph:
    def __init__(self, graph: Dict | str = None):
        self.graph = {}
        if isinstance(graph, str):
            self.load(graph)
        elif isinstance(graph, dict):
            self.graph = graph

    def get_graph(self) -> Dict:
        return self.graph
    
    def load(self, graph: Dict | str) -> Dict:
        if isinstance(graph, str):
            try:
                with open(graph, "r") as f:
                    self.graph = json.load(f)
            except Exception as e:
                logger.error(f"Error loading graph from file {graph}: {e}")
                raise
        elif isinstance(graph, dict):
            self.graph = graph
        else:
            raise ValueError("Graph must be a dictionary or a path to a JSON file.")

        return self.graph
    
    def find_node(self, node_id: Optional[str] = None, module: Optional[str] = None, action: Optional[str] = None) -> list:
        nodes = self.graph.get('nodes', {})
        results = []

        if node_id:
            if node_id in nodes:
                node = nodes[node_id].copy()
                node['id'] = node_id
                return [node]
            else:
                return []

        if not module and not action:
            return []

        for nid, node_data in nodes.items():
            if module is not None and not module.startswith(('modules.', 'custom.')):
                module = f"modules.{module}"

            match_module = (not module) or (node_data.get('module') == module)
            match_action = (not action) or (node_data.get('action') == action)
            
            if match_module and match_action:
                node = node_data.copy()
                node['id'] = nid
                results.append(node)
                
        return results
    
    def update_value(self, node_id: str, param_key: str, new_value: Any) -> Dict:
        node = self.graph.get('nodes', {}).get(node_id)
        if not node:
            logger.warning(f"Node with id '{node_id}' not found in graph.")
            return

        param = node.get('params', {}).get(param_key)
        if param is None:
            logger.warning(f"Param '{param_key}' not found in node '{node_id}'.")
            return

        if 'value' in param:
            param['value'] = new_value
        elif not param: # is an empty dict {}
            param.update({'value': new_value})
        else:
            logger.warning(f"Cannot update param '{param_key}' in node '{node_id}': 'value' key is missing and dict is not empty.")

        return self.graph

