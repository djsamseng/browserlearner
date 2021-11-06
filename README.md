
### Setting up audio loopback
```bash
$ pacmd load-module module-loopback latency_msec=5pav
$ pavucontrol 
# Input Devices
# Show All Input Devices
# Click the green check mark next to "Monitor of Built-in Audio Analog Stereo"
```

```python3
import pocketsphinx
for phrase in pocketsphinx.LiveSpeach():
    print(phrase)
```

Now play some sound (ex: ./data/recordings/a.wav) and it should print out what it heard

### Recording new sounds

```bash
$ python3 record.py hello # saves to ./data/recordings/hello.wav
```

If you run into "OSError: [Errno -9998] Invalid number of channels"
and it should work, try a few more times and eventually it works.
Additionally you can run `$ sudo killall pulseaudio`

If you run into "OSError: [Errno -9981] Input overflowed"
this may be because the microphone volume is too high. Turning down
the microphone gain may solve this issue

## Install

### Pocketsphinx
1. Download and install sphinxbase and pocketsphinx per [instructions](https://cmusphinx.github.io/wiki/tutorialpocketsphinx/#installation-on-unix-system)
  1. `sudo apt-get install python-dev` to fix "Could not link test program to Python. Maybe the main Python library has been
installed in some non-standard library path. If so, pass it to configure,
via the LDFLAGS environment variable."

### pycli
```bash
$ cd pycli
$ python -m venv env
$ source ./env/bin/activate
$ pip install -r requirements.txt
```

```bash
$ cd ./server && yarn install
```

## Run
```bash
$ cd ./server && yarn start
```

```bash
$ cd ./client && yarn start
# open localhost:3000 in Chrome
```

```bash
$ cd pycli
$ source ./env/bin/activate
$ python3 main.py --letters
$ python3 main.py --youtube
```