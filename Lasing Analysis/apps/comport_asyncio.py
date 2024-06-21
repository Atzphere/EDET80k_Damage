import asyncio
import serial_asyncio
import serial
import logging
import time
from typing import Tuple

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

DEFAULT_EOL_CHAR = b'\r' # programs such as PIX connect need this
DEBUG = False

'''
TODO: make query handle ports that aren't actually functional
'''

class COMInterfaceException(Exception):
    pass

class IncompleteResponseException(Exception):
    pass

class NoResponseException(IncompleteResponseException):
    pass

class QueryFailedException(Exception):
    pass

class COMInterface(object):

    def __init__(self, com_port):
        self.com_port = com_port
        self.io_lock = asyncio.Lock()

    async def query_port(self, queries, **kwargs):
        '''
        Attempts to query the port for information, wait a certain amount,
        then read the port for information.

        This is ASYNCHRONOUS: it must be run with through asyncio, i.e.
        asyncio.run(), asyncio.gather(). This allows OptrisHandler to process
        many requests concurrently without blocking.

        param queries: str or List[str]: The query string(s), sent one by one.
                                         Do not include EOL characters here.

        '''
        try:
            async with self.io_lock:
                responses = await query_port(self.com_port, queries, **kwargs)
        except IncompleteResponseException as e:
            raise
        except Exception as e:
            raise COMInterfaceException("COM Query failed") from e
        return responses
    
    def query_port_blocking(self, queries, **kwargs):
        return asyncio.run(self.query_port(queries, **kwargs))

async def query_port(port, commands, tolerant=False, **kwargs):
    '''
    Queries a given COM port with one or more commands

    params:

    port: str : The port to connect and interact with.

    commands: str or List[Str]: The query(s) to send to the port. Order is respected.

    tolerant: bool default False: Whether or not to raise an exception upon a dropped response.
                                  Returns false on failure if true. Use to handle faults externally

    kwargs passed to other functions: use_optris_workaround: use to deal with Pix Connect dropping every 3rd request.
    
    returns: str or List[str]: list of responses from the serial device.

    '''
    query_start = time.time()
    timed_out = False
    reader, writer = await open_com_connection(port, **kwargs)
    logging.debug(f"opening COM connection took {time.time() - query_start} seconds.")
    logging.debug(f"Port {port} initialized.")
    try:
        start = time.time()
        messages = commands
        if type(messages) == str:
            message_count = 1
        else:
            message_count = len(messages)

        await send_commands(writer, messages, **kwargs)
        logging.debug(f"sending took {time.time() - start} seconds.")

        start = time.time()
        try:
            received = await read(reader, message_count, tolerant=tolerant, **kwargs)
            logging.debug(f"receiving took {time.time() - start} seconds.")
        except (IncompleteResponseException, NoResponseException) as e: # error catching flag
            timed_out = True
            received = False

    finally: # try to shutdown the writer cleanly
        writer.close()
        start = time.time()
        if timed_out and tolerant:
            logging.warn("One or more queries timed out")
        elif timed_out and not tolerant:
            raise QueryFailedException("Query failed to get valid response")
        else:
            await writer.wait_closed() # this hangs if there was a timeout during the query
            logging.debug(f"closing took {time.time() - start} seconds.")
            logging.debug(f"Port {port} closed.")
        logging.info(f"Query took {time.time() - query_start} seconds. ({(time.time() - query_start) / message_count:.4f}s per request)")
    return received


async def open_com_connection(port, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    '''
    Initializes a COM connection, returns StreamReader and StreamWriter objects
    '''
    reader, writer = await serial_asyncio.open_serial_connection(url=port, baudrate=baudrate, bytesize=bytesize, parity=parity, stopbits=stopbits)
    return reader, writer


async def send_commands(w, msgs, delay=0.04, eol_character=DEFAULT_EOL_CHAR, optris_workaround=True):
    if type(msgs) == str:
        cmd = msgs.encode() + DEFAULT_EOL_CHAR
        w.write(cmd)
        logging.debug(f'sent: {cmd.decode().rstrip()}')
        await asyncio.sleep(delay)
    else:
        for count, msg in enumerate(msgs):
            cmd = msg.encode() + DEFAULT_EOL_CHAR
            w.write(cmd)
            logging.debug(f'sent: {cmd.decode().rstrip()}')
            if count == 2 and optris_workaround:
                await asyncio.sleep(delay * 1.25) # works better with optris which likes to otherwise miss the 3rd message
            else:
                await asyncio.sleep(delay)
            await w.drain()
    logging.debug('Done sending')


async def read(r, expected_msgs=1, readline_eol_character=DEFAULT_EOL_CHAR, tolerant=False, optris_workaround=True, timeout=0.4):
    messages = []
    logging.debug(f"Expecting {expected_msgs} messages.")
    for i in range(expected_msgs):
        try:
            if i == 2 and optris_workaround:
                msg = await asyncio.wait_for(r.readuntil(readline_eol_character), timeout=timeout)
            else:
                msg = await asyncio.wait_for(r.readuntil(readline_eol_character), timeout=timeout)
            logging.debug(f'received: {msg}')
            messages.append(msg)
        except asyncio.exceptions.TimeoutError as e:
            if len(messages) == expected_msgs - 1 and expected_msgs > 1 and optris_workaround:
                logging.debug(f'one message out of {len(messages)} timed out, janking')
                messages.insert(2, "timed out")
                break
            elif tolerant and len(messages) > 0:
                logging.warn(f'one message out of {len(messages)} timed out')
                messages.append("timed out")
            elif len(messages) > 0:
                raise IncompleteResponseException(f"Reading serial port timed out: Not all queries were returned.") from e
            else:
                raise NoResponseException("Reading serial port timed out: No response received.") from e
    logging.debug("Done receiving")

    if len(messages) == 1:
        return messages[0]
    else:
        return messages


if __name__ == "__main__":
    logging.debug(asyncio.run(query_port(
        "COM4", ['?T', '!ImgTemp', '?AreaLoc(0)', 'qux'])))
