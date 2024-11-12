import comport_asyncio
import time
import pyautogui
from tqdm import tqdm

# print(pyautogui.KEY_NAMES)

pixconnect = comport_asyncio.COMInterface('COM4')

frame_rate = 27 # hz
sample_time = 45 # seconds
frame_num = sample_time * frame_rate

sendung = "!Snapshot"

time.sleep(5)

for i in tqdm(range(frame_num)):
    try:
        pixconnect.query_port_blocking(sendung)
    except comport_asyncio.COMInterfaceException as e:
        pass
    time.sleep(0.1)
    pyautogui.press('right')

    
    