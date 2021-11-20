
import argparse
import base64

import asyncio
import cv2
import numpy as np
import pocketsphinx
import socketio


sio = socketio.AsyncClient()

DIC_PATH = "/home/samuel/dev/browserlearner/pycli/cmudict-en-us.dict"
SERVER_URL = "http://localhost:4000"
BROWSER_WIDTH = 600
BROWSER_HEIGHT = 800

def get_line_len(line, size):
    # 5 = 80a + b
    # 1 = 6a + b
    size_mult = 6 * size / 74 + 38 / 74
    return len(line) * size_mult

async def train_letters():
    letters = []
    # ! # etc.
    for i in range(33, 47+1):
        pass
        #letters.append(chr(i))
    # 0 to 9
    for i in range(48, 57+1):
        pass
        #letters.append(chr(i))
    # : ; etc.
    for i in range(58, 65+1):
        pass
        #letters.append(chr(i))
    # A to Z
    for i in range(65, 90+1):
        pass
        #letters.append(chr(i))
    # [ ^ etc
    for i in range(91, 97+1):
        pass
        #letters.append(chr(i))
    # a to z
    for i in range(97, 122):
        letters.append(chr(i))
    # { | etc.
    for i in range(122, 126+1):
        pass
        #letters.append(chr(i))

    for letter in letters:
        size = np.random.randint(6, 80)
        line_len = get_line_len(letter, size)
        x = np.random.randint(BROWSER_WIDTH - line_len)
        y = np.random.randint(BROWSER_HEIGHT - 400 - line_len)

        rotate = np.random.randint(-40, 40)
        url = SERVER_URL + \
            "/letters?letter=" + \
            letter + \
            "&x=" + str(x) + "&y=" + str(y) + \
            "&size=" + str(size) + \
            "&rotate=" + str(rotate)
        await goto(url)
        print("url:", url)
        for phrase in pocketsphinx.LiveSpeech(dic=DIC_PATH):
            phrase_str = str(phrase).rstrip(".")
            print(phrase, phrase.score())
            if phrase_str == letter and phrase.score() > -1800:
                print("Correct!", letter)
                break

async def train_youtube():
    await asyncio.sleep(2)
    await goto("https://www.youtube.com/watch?v=n068fel-W9I")
    await asyncio.sleep(2)
    await sio.emit("mouseMove", {
        "x": 280,
        "y": 290
    })
    await sio.emit("mouseClick", {
        "x": 380,
        "y": 290
    })

@sio.on("frame")
async def on_frame(data):
    #data = base64.b64encode(data)
    data = base64.b64decode(data)
    data = np.frombuffer(data, dtype=np.uint8)
    data = cv2.imdecode(data, flags=cv2.IMREAD_COLOR)
    cv2.imshow("frame", data)
    cv2.waitKey(5)

async def emit(event_name):
    ft = asyncio.get_event_loop().create_future()
    def cb():
        ft.set_result(True)
    await sio.emit(event_name, callback=cb)
    await ft

async def goto(url):
    ft = asyncio.get_event_loop().create_future()
    def cb():
        ft.set_result(True)
    await sio.emit("goto", url, callback=cb)
    await ft
    #headers = {'content-type': 'application/json'}
    #r = requests.post(SERVER_URL + "/goto",
    #   headers=headers, data=json.dumps({ "url": url }))
    #print(r.status_code, r.reason)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--youtube",
        dest="youtube",
        nargs="?",
        const=True, # if provided set to true
        default=False # if not provided set to false
    )
    parser.add_argument(
        "--letters",
        dest="letters",
        nargs="?",
        const=True,
        default=False,
    )
    return parser.parse_args()

async def main():
    args = parse_args()
    print(args)
    await sio.connect(SERVER_URL)
    await emit("startWebBrowser")
    if args.youtube:
        await train_youtube()
    elif args.letters:
        await train_letters()
    else:
        print("No option specified:", args)

async def close():
    await asyncio.sleep(0.1)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            main()
        )
    except KeyboardInterrupt as e:
        print("Keyboard Interrupt")
    finally:
        print("Cleaning up")
        loop.run_until_complete(
            close()
        )

        print("Exiting")
