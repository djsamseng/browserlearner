
import time

import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import scipy.signal
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
            "step size ms:", jump_ms,
            "chunk size:", chunk_size
        )
        ### FFT of how the frequencies change instead of absolute difference

        num_steps = len(range(0, audio.shape[0] - chunk_size, step_size))
        time_buckets = np.zeros((num_steps, chunk_size), dtype=np.complex128)
        x = np.linspace(0, rate, chunk_size)
        for i in range(num_steps):
            audio_idx = i * step_size
            audio_in = audio[audio_idx:audio_idx+chunk_size]
            # Use the magnitude of the frequency
            # Phase information is dropped
            buckets = np.abs(np.fft.fft(audio_in))
            buckets[np.where(x<50)] = 0
            buckets[np.where(x>1500)] = 0
            # TODO: smooth out the buckets?
            # TODO: Hello at different frequencies still sounds the same
            freq_max = np.max(buckets)
            inc_factor = 7.0 / freq_max
            buckets[:] *= inc_factor
            #buckets[np.abs(buckets<1)] = 0
            #has some zero error jump_ms=5
            #buckets[np.where(x<100)] = 0
            #buckets[np.where(x>rate-1000)] = 0
            #buckets[np.abs(buckets<2)] = 0
            if False:
                print(i/num_steps)
                plt.ylim(0, 7)
                plt.xlim(0,8000)
                plt.plot(x, np.abs(buckets))
                plt.show()
            time_buckets[i] = buckets

        self.audio = audio
        self.time_buckets = time_buckets
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.rate = rate
        self.step_size = step_size

    def find_closest_fft_bucket(self, other_buckets):
        # instead of norm, find patterns in how the frequencies change
        # hello at different frequencies still sounds the same
        closest_distance = np.linalg.norm(self.time_buckets[0] - other_buckets)
        closest_idx = 0
        for i in range(self.num_steps):
            distance = np.linalg.norm(self.time_buckets[i] - other_buckets)
            if distance < closest_distance:
                closest_distance = distance
                closest_idx = i
        print(closest_distance)
        return closest_idx

    def create_closest_match_fft(self, goal_fft_anaylzer):
        other = goal_fft_anaylzer
        np.testing.assert_allclose(self.step_size, other.step_size)
        np.testing.assert_allclose(self.rate, other.rate)
        output = np.zeros(shape=other.audio.shape)
        for i in range(other.num_steps):
            goal_buckets = other.time_buckets[i]
            closest_idx = self.find_closest_fft_bucket(goal_buckets)
            closest_audio_idx = closest_idx * self.step_size
            # TODO: Is chunk_size the right length?
            closest_audio = self.audio[closest_audio_idx:closest_audio_idx+self.chunk_size]
            output_audio_idx = i * self.step_size
            output[output_audio_idx:output_audio_idx+self.chunk_size] = closest_audio
        return output.astype(np.float32)

    def find_closest_audio_chunk(self, other_chunk):
        audio_len = min(self.chunk_size, other_chunk.shape[0])
        closest_distance = np.linalg.norm(self.audio[0:audio_len] - other_chunk)
        closest_idx = 0
        for i in range(0, self.audio.shape[0] - audio_len, audio_len):
            distance = np.linalg.norm(self.audio[i:i+audio_len] - other_chunk)
            if distance < closest_distance:
                closest_distance = distance
                closest_idx = i
        print(closest_distance)
        return closest_idx

    def create_closest_match_audio(self, goal_fft_analyzer):
        other = goal_fft_analyzer
        np.testing.assert_allclose(self.rate, other.rate)
        np.testing.assert_allclose(self.chunk_size, other.chunk_size)
        output = np.zeros(shape=other.audio.shape)
        for i in range(0, other.audio.shape[0], other.chunk_size):
            goal_audio = other.audio[i:i+other.chunk_size]
            closest_audio_idx = self.find_closest_audio_chunk(goal_audio)
            audio_len = min(self.chunk_size, goal_audio.shape[0])
            output[i:i+audio_len] = self.audio[closest_audio_idx:closest_audio_idx+audio_len]
        return output.astype(np.float32)

def change_audio_frequency(audio, freq_change=-1, chunk_size=1024):
    aud = np.zeros_like(audio)
    for i in range(0, audio.shape[0] - chunk_size, chunk_size):
        buckets = np.fft.fft(audio[i:i+chunk_size])
        buckets = np.roll(buckets, freq_change)
        if freq_change > 0:
            buckets[:freq_change] = 0
        else:
            buckets[freq_change:] = 0
        back = np.fft.ifft(buckets)
        aud[i:i+chunk_size] = back.real.astype(np.float32)
    return aud

def speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]


def stretch(sound_array, f, window_size, h):
    """ Stretches the sound by a factor `f` """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( int(len(sound_array) /f + window_size), dtype=np.complex128)

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):
        i = int(i)
        # two potentially overlapping subarrays
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # add to result
        i2 = int(i/f)
        result[i2 : i2 + window_size] += hanning_window*a2_rephased

    #result = ((2**(16-4)) * result/result.max()) # normalize (16bit)
    return result.real.astype(np.float32)

def pitch_shift(sound_array, factor=1.0, window_size=2**13, h=2**11):
    stretched = stretch(sound_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)


def plot_test_recordings():
    path_begin = "./data/recordings/test"
    rate = 44100
    paths = [path_begin + str(i) + ".wav" for i in range(1,6,1)]
    loads = [librosa.load(path, sr=rate) for path in paths]
    audios = [audio for audio, _ in loads]
    num_plots = len(audios)

    chunk_size = 1024
    step_size = 1024//4
    audio_lens = [len(aud) for aud in audios]
    audio_len = min(audio_lens)
    x = np.linspace(0, rate, chunk_size)
    for i in range(0, audio_len - chunk_size, step_size):
        print("step:", i / audio_len)
        for j in range(num_plots):
            freq_buckets = np.fft.fft(audios[j][i:i+chunk_size])
            plt.subplot(num_plots, 1, j+1)
            plt.ylim(0, 7)
            plt.xlim(50,200)
            plt.plot(x, np.abs(freq_buckets))
        plt.show()

def repeat_after_me():
    rate = 16000

    hand_test_audio_file_name = "./data/recordings/test_from_phonemic_chart_sounds.wav"
    hand_test_audio, _ = librosa.load(hand_test_audio_file_name, sr=rate)

    word = "TEST"
    file_name = "./data/recordings/" + word.lower() + ".wav"
    audio, _ = librosa.load(file_name, sr=rate)
    if False:
        play_numpy(audio)
        audio_shifted = librosa.effects.pitch_shift(audio, sr=16000, n_steps=-8)
        play_numpy(audio_shifted)
        play_numpy(change_audio_frequency(audio, 500))
        time.sleep(1)
        play_numpy(pitch_shift(audio, 1.0, 1024, 1024//2))
        time.sleep(1)
        for i in np.arange(0.7, 1.4, 0.1):
            shifted = pitch_shift(audio, i, 1024, 1024//2)
            print(shifted.shape)
            play_numpy(shifted)


    hand_test_fft_analyzer = FFTAnalyzer(hand_test_audio, rate)

    test_fft_analyzer = FFTAnalyzer(audio, rate)
    phonemics_file_name = "./data/recordings/phonemic_chart_sounds.wav"
    phonemics_audio, _ = librosa.load(phonemics_file_name, sr=rate)
    phonemics_fft_analyzer = FFTAnalyzer(phonemics_audio, rate)



    if False:
        hand_match_audio = phonemics_fft_analyzer.create_closest_match_audio(hand_test_fft_analyzer)
        print(hand_match_audio, hand_match_audio.dtype)
        play_numpy(hand_test_fft_analyzer.audio)
        time.sleep(1)
        play_numpy(hand_match_audio)
    if False:
        audio_match = phonemics_fft_analyzer.create_closest_match_audio(test_fft_analyzer)
        play_numpy(audio_match)
        time.sleep(1)
        play_numpy(test_fft_analyzer.audio)
        return

    hand_match = phonemics_fft_analyzer.create_closest_match_fft(hand_test_fft_analyzer)
    play_numpy(hand_match, rate=rate)
    time.sleep(1)
    play_numpy(hand_test_audio, rate=rate)

    # now match_audio and hand_match sound similar
    match_audio = phonemics_fft_analyzer.create_closest_match_fft(test_fft_analyzer)
    print(match_audio.dtype, match_audio.shape, audio.shape)
    time.sleep(1)
    play_numpy(match_audio, rate=rate)
    time.sleep(1)
    play_numpy(audio, rate=rate)

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
        input_values = tokenizer(input_audio, return_tensors="pt").input_values
        input_values = input_values.to(device)
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        return transcription

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

def find_peaks(audio, min_step_size):
    pass

def get_distances_between_elements(ar):
    distances = []
    for i in range(1, len(ar), 1):
        distances.append(ar[i] - ar[i-1])
    return distances

def make_e_of_test():
    file_name = "./data/recordings/test3.wav"
    input_audio, _ = librosa.load(file_name, sr=16000)
    plt.subplot(3, 1, 1)
    #audio_sample = input_audio[6280:6380]
    #print(np.where(audio_sample == np.max(audio_sample)))
    start = 6302
    end = start + 3200
    audio_sample = input_audio[start:end]
    print(np.where(audio_sample == np.max(audio_sample[200:])))
    peaks_all, _ = scipy.signal.find_peaks(audio_sample, prominence=0.2)
    peaks = np.concatenate([peaks_all[:12], [peaks_all[12], peaks_all[14], peaks_all[16]]])

    peak_distances = get_distances_between_elements(peaks)
    print(peaks, peak_distances)

    second_half_peaks, _ = scipy.signal.find_peaks(audio_sample[1400:], prominence=0.05)
    second_half_peaks += 1400
    print(second_half_peaks, get_distances_between_elements(second_half_peaks))
    markevery = np.concatenate([peaks, second_half_peaks])
    plt.plot(audio_sample, "-bD", markevery=markevery)

    data = np.zeros_like(audio_sample)
    minimal_sample = audio_sample[:78]
    # tile minimal_sample 15 times, spreading out the signal by 2 frames each time
    # then tile 20 times
    plt.subplot(3, 1, 2)
    plt.plot(minimal_sample)

    num_repeats = 5
    copy_play = np.tile(audio_sample, num_repeats)
    dist_to_increase = audio_sample[-1] - audio_sample[0]
    print(dist_to_increase)
    for i in range(num_repeats):
        audio_start = i * len(audio_sample)
        audio_end = (i + 1) * len(audio_sample)
        #copy_play[audio_start:audio_end] += dist_to_increase * i

    plt.subplot(3, 1, 3)
    plt.plot(copy_play)
    plt.show()

    play_numpy(copy_play)


if __name__ == "__main__":
    make_e_of_test()
    #plot_test_recordings()
    #repeat_after_me()
    #wav2vec_test()