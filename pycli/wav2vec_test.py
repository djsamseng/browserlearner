
import time

import torch
import librosa
import matplotlib.pyplot as plt
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
                        #input_device_index=0, # set default input source instead
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
        # TODO return array of intervals
        return frames

class FFTAnalyzer():
    def __init__(self, audio, rate, jump_ms = 20) -> None:
        chunk_size = 1024
        windows_per_second = rate / chunk_size
        fft_sampled_every_n_ms = 1000 / windows_per_second
        # 64 ms for rate=16000
        bucket_resolution = rate / chunk_size
        step_size = int(rate * jump_ms // 1000)
        print(
            "FFT window size:", fft_sampled_every_n_ms, "ms",
            "FFT resoltion:", bucket_resolution, "hz",
            "step size:", step_size,
            "chunk size:", chunk_size
        )

        num_steps = len(range(0, audio.shape[0] - chunk_size, step_size))
        time_buckets = np.zeros((num_steps, chunk_size), dtype=np.complex128)
        x = np.linspace(0, rate, chunk_size)
        for i in range(num_steps):
            audio_idx = i * step_size
            audio_in = audio[audio_idx:audio_idx+chunk_size]
            buckets = np.fft.fft(audio_in)
            buckets[np.where(x<100)] = 0
            buckets[np.where(x>rate-1000)] = 0
            buckets[np.abs(buckets<1)] = 0
            #has some zero error jump_ms=5
            #buckets[np.where(x<100)] = 0
            #buckets[np.where(x>rate-1000)] = 0
            #buckets[np.abs(buckets<2)] = 0
            #plt.plot(x, np.abs(buckets))
            #plt.show()
            time_buckets[i] = buckets

        self.audio = audio
        self.time_buckets = time_buckets
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.rate = rate
        self.step_size = step_size

    def find_closest_fft_bucket(self, other_buckets):
        closest_distance = np.linalg.norm(self.time_buckets[0] - other_buckets)
        closest_idx = 0
        for i in range(self.num_steps):
            distance = np.linalg.norm(self.time_buckets[i] - other_buckets)
            if distance < closest_distance:
                closest_distance = distance
                closest_idx = i
        print(closest_distance)
        return closest_idx

    def create_closest_match(self, goal_fft_anaylzer):
        other = goal_fft_anaylzer
        np.testing.assert_allclose(self.step_size, other.step_size)
        np.testing.assert_allclose(self.rate, other.rate)
        output = np.zeros(shape=other.audio.shape)
        for i in range(other.num_steps):
            goal_buckets = other.time_buckets[i]
            closest_idx = self.find_closest_fft_bucket(goal_buckets)
            closest_audio_idx = closest_idx * self.step_size
            closest_audio = self.audio[closest_audio_idx:closest_audio_idx+self.chunk_size]
            output_audio_idx = i * self.step_size
            output[output_audio_idx:output_audio_idx+self.chunk_size] = closest_audio
        return output.astype(np.float32)


def repeat_after_me():
    rate = 16000
    hand_test_audio_file_name = "./data/recordings/test_from_phonemic_chart_sounds.wav"
    hand_test_audio, _ = librosa.load(hand_test_audio_file_name, sr=rate)
    hand_test_fft_analyzer = FFTAnalyzer(hand_test_audio, rate)

    word = "TEST"
    file_name = "./data/recordings/" + word.lower() + ".wav"

    audio, _ = librosa.load(file_name, sr=rate)
    test_fft_analyzer = FFTAnalyzer(audio, rate)
    phonemics_file_name = "./data/recordings/phonemic_chart_sounds.wav"
    phonemics_audio, _ = librosa.load(phonemics_file_name, sr=rate)
    phonemics_fft_analyzer = FFTAnalyzer(phonemics_audio, rate)


    # TODO: This should be able to find an exact match, even step_size=1 has 12.744 error
    # thus frequency analysis doesn't provide the signal
    hand_match = phonemics_fft_analyzer.create_closest_match(hand_test_fft_analyzer)
    #play_numpy(hand_test_audio, rate=rate)
    play_numpy(hand_match, rate=rate)

    # now match_audio and hand_match sound similar
    match_audio = phonemics_fft_analyzer.create_closest_match(test_fft_analyzer)
    print(match_audio.dtype, match_audio.shape, audio.shape)
    play_numpy(audio, rate=rate)
    time.sleep(1)
    play_numpy(match_audio, rate=rate)

def play_numpy(data, rate=RATE):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=rate,
        output=True,
    )
    data_bytes = data.tobytes()
    stream.write(data_bytes)

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


    def binary_search_remove(audio_orig, actual, from_begin, x0=None, x1=None, ):
        if x0 is None:
            x0 = 0
        if x1 is None:
            x1 = audio_orig.shape[0]
        itr = int((x1 - x0) // 2) + x0

        if from_begin:
            audio = audio_orig[itr:]
        else:
            audio = audio_orig[:itr]

        transcription = predict_audio(audio)
        print(x0, x1, itr, transcription)
        if from_begin:
            if transcription == actual:
                if itr == x0:
                    return itr
                return binary_search_remove(audio_orig, actual, from_begin=from_begin, x0=itr, x1=x1)
            if itr == x0:
                return itr - 1
            return binary_search_remove(audio_orig, actual, from_begin=from_begin, x0=x0, x1=itr)
        else:
            if transcription == actual:
                if itr == x0:
                    return itr
                return binary_search_remove(audio_orig, actual, from_begin=from_begin, x0=x0, x1=itr)
            if itr == x0:
                return itr + 1
            return binary_search_remove(audio_orig, actual, from_begin=from_begin, x0=itr, x1=x1)

    def find_mininal_word(word="TEST"):
        file_name = "./data/recordings/" + word.lower() + ".wav"
        data = wavfile.read(file_name)
        framerate = data[0]
        sounddata = data[1]
        input_audio_orig, _ = librosa.load(file_name, sr=16000)
        input_audio = input_audio_orig.copy()
        print(predict_audio(input_audio[18038:]), predict_audio(input_audio[:29590]))

        begin_idx = binary_search_remove(input_audio_orig, word, from_begin=True)
        input_audio = input_audio[begin_idx:]

        end_idx = binary_search_remove(input_audio_orig, word, from_begin=False)
        input_audio = input_audio[:end_idx]

        transcription = predict_audio(input_audio)
        print(input_audio.shape, "Begin idx:", begin_idx, "End idx:", end_idx, "T:", transcription)
        play_numpy(input_audio)
        play_numpy(input_audio_orig)
        play_numpy(input_audio)

    find_mininal_word("TEST")
    find_mininal_word("A")
    return
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
    repeat_after_me()
    #wav2vec_test()