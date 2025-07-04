from mellon.config import CONFIG, ColorCodes
import logging
import asyncio
logger = logging.getLogger('mellon')

from modules import MODULE_MAP
from mellon.server import server

# welcome message
logger.info(f"""{ColorCodes.BLUE}
╭──────────────────────╮
│  Welcome to Mellon!  │
╰──────────────────────╯
Speak Friend and Enter: {CONFIG.server['scheme']}://{CONFIG.server['ip']}:{CONFIG.server['port']}""")

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(server.run())
    loop.run_forever()
except KeyboardInterrupt:
    print('')
    logger.info(f"Received keyboard interrupt. Exiting...")
except Exception as e:
    logger.error(e)
    raise e
finally:
    loop.run_until_complete(server.cleanup())
    tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if tasks:
        logger.debug(f"Cancelling {len(tasks)} outstanding tasks. This might take a few seconds...")
        for task in tasks:
            # give some info about the task
            logger.debug(f"Task {task} is {task.get_name()}")
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.close()
    logger.info(f"{ColorCodes.BLUE}Namárië!")