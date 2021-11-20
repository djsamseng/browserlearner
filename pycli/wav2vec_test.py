
import time

import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000 #44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

class AudioGrabber():
    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        #output=True,
                        #input_device_index=0,
                        frames_per_buffer=CHUNK)
        self.interval = 0.1
        self.predict_seconds = 5


    def grab_data(self):

        frames = []

        for i in range(0, int(RATE / CHUNK * self.predict_seconds)):
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            if data:
                frames.append(np.frombuffer(data, dtype=np.float32))
            else:
                print("No data")
        frames = np.array(frames)
        frames = np.concatenate(frames)
        return frames

def wav2vec_test():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model = model.to(device)

    def predict_audio(input_audio):
        t0 = time.time()
        input_values = tokenizer(input_audio, return_tensors="pt").input_values
        input_values = input_values.to(device)
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        t1 = time.time()
        return transcription
        print(transcription, " took:", t1 - t0)

    def read_and_predict(file_name):
        t0 = time.time()

        data = wavfile.read(file_name)
        framerate = data[0]
        sounddata = data[1]
        time_ar = np.arange(0,len(sounddata))/framerate
        input_audio, _ = librosa.load(file_name, sr=16000)
        transcription = predict_audio(input_audio)
        t1 = time.time()
        print(transcription, input_audio.shape, input_audio.dtype, "total:", t1 - t0)

    audio_grabber = AudioGrabber()
    print("Start talking")
    while True:
        input_audio = audio_grabber.grab_data()
        transcription = predict_audio(input_audio)
        # a = EH
        # b = B
        # c = SEE
        # d = D
        # e = E
        # f = F
        # g = GE
        # h = H
        # i = AY
        # j = J
        # k = K
        # l = L
        # m = M
        # n = N
        # o = OH
        # p = P
        # q = QU
        # r = HAR/AR
        # s = S
        # t = T
        # u = YOU
        # v = VEY/VE/V
        # w = W
        # x = X
        # y = WHY
        # z = IZZ E/ZI/Z
        print("Transcription:", transcription)


    file_name = "./data/recordings/test.wav"
    read_and_predict(file_name)
    file_name = "./data/recordings/a.wav"
    read_and_predict(file_name)
    file_name = "./data/recordings/test.wav"
    read_and_predict(file_name)
    file_name = "./data/recordings/a.wav"
    read_and_predict(file_name)
    file_name = "./data/recordings/test.wav"
    read_and_predict(file_name)
    file_name = "./data/recordings/a.wav"
    read_and_predict(file_name)

if __name__ == "__main__":
    wav2vec_test()