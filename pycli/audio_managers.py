
import wave

import pyaudio


CHUNK = 1024 * 2
CHANNELS = 2
FORMAT = pyaudio.paInt16
RATE = 44100

RECORDINGS_DIR = "./data/recordings/"

class AudioManager():
    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()

    def open_stream(self):
        self.stream = self.p.open(
            rate=RATE,
            format=FORMAT,
            channels=CHANNELS,
            output=True
        )

    def print_output_devices(self):
        info = self.p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        for i in range(0, num_devices):
            device_data = self.p.get_device_info_by_host_api_device_index(0, i)
            device_name = device_data.get('name')
            max_output_channels = device_data.get("maxOutputChannels")
            print("Device:", device_name, "device_idx:", i, "Max output channels:", max_output_channels)

    def tick(self, play_chunk=None):
        if play_chunk is not None:
            self.stream.write(play_chunk, CHUNK)

def main():
    audio_manager = AudioManager()
    audio_manager.print_output_devices()
    
    audio_manager.open_stream()

    wf = wave.open(RECORDINGS_DIR + "a.wav", 'rb')
    data = wf.readframes(CHUNK)
    while data != b"":
        audio_manager.tick(data)
        data = wf.readframes(CHUNK)

if __name__ == "__main__":
    main()