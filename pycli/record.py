
import argparse
import wave
import sys
import time

import pyaudio
import pynput

CHUNK = 1024 * 2
CHANNELS = 2
FORMAT = pyaudio.paInt16
RATE = 44100
WAVE_OUTPUT_DIR = "./data/recordings/"

class KeypressRecorder():
    def __init__(self) -> None:
        self.__do_record = True

    def get_input_devices(self):
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")
        for i in range(0, numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            max_input_channels =device_info.get("maxInputChannels")
            
            if max_input_channels > 0:
                print(
                    "Idx:", i, 
                    device_info.get("name"), 
                    "max_input_channels", max_input_channels,
                    "default sample rate:", device_info.get("defaultSampleRate"))

        p.terminate()

    def record_until_press(self, filename):
        def on_press(key):
            if key == pynput.keyboard.KeyCode.from_char("r"):
                self.__do_record = False
        self.keyboard_listener = pynput.keyboard.Listener(
            on_press=on_press
        )
        self.keyboard_listener.start()
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            # sudo killall pulseaudio
            input_device_index=0,
            frames_per_buffer=CHUNK,
        )

        print("Recording: press r to stop")
        frames = []
        while self.__do_record:
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("Finished recording. Num frames:", len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()

        full_filename = WAVE_OUTPUT_DIR + filename + ".wav"
        wf = wave.open(full_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print("Saved to:", full_filename)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    return parser.parse_args()

def main():
    args = parse_args()
    keypress_recorder = KeypressRecorder()
    keypress_recorder.get_input_devices()
    keypress_recorder.record_until_press(args.filename)

if __name__ == "__main__":
    main()

