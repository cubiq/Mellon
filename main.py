from mellon.config import CONFIG, ColorCodes
import logging
import asyncio
import signal

logger = logging.getLogger('mellon')

from modules import MODULE_MAP
from mellon.modelstore import modelstore
from mellon.server import server

# welcome message
logger.info(f"""{ColorCodes.BLUE}
╭──────────────────────╮
│  Welcome to Mellon!  │
╰──────────────────────╯
Speak Friend and Enter: {CONFIG.server['scheme']}://{CONFIG.server['ip']}:{CONFIG.server['port']}""")

async def main():
    await server.run()
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        logger.info("Shutdown initiated. Waiting for server to cleanup...")
    finally:
        logger.info("If there are any outstanding tasks, this might take a few seconds.")
        await server.cleanup()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task = loop.create_task(main())

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, main_task.cancel)
    except NotImplementedError:
        # for windows
        pass

    try:
        loop.run_until_complete(main_task)
    finally:
        loop.close()
        logger.info(f"{ColorCodes.BLUE}Namárië!")
