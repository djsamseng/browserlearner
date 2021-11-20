
import argparse
import wave

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import scipy.fft
from scipy.io import wavfile

CHUNK = 1024 * 2
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 44100
WAVE_OUTPUT_DIR = "./data/recordings/"

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


def fft_subtract(test, background, rate, plot=False):
    background_freq_buckets = np.fft.fft(background)
    test_freq_buckets = np.fft.fft(test)
    filtered_buckets = np.abs(test_freq_buckets) - np.abs(background_freq_buckets)
    #filtered_buckets[filtered_buckets < 0] = 0
    #filtered_buckets[test_freq_buckets < 0] = -filtered_buckets[test_freq_buckets < 0]
    if plot:
        x = np.linspace(0, rate, len(background_freq_buckets))
        ylim = 5 * 10 ** 5
        xlim = 44100
        plt.subplot(3, 1, 1)
        plt.ylim(-ylim, ylim)
        plt.xlim(0, 2000)
        plt.plot(x, test_freq_buckets)
        plt.subplot(3, 1, 2)
        plt.ylim(-ylim, ylim)
        plt.xlim(0, 2000)
        plt.plot(x, background_freq_buckets)
        plt.subplot(3, 1, 3)
        plt.ylim(-ylim, ylim)
        plt.xlim(0, 2000)
        plt.plot(x, filtered_buckets)

        plt.show()
    return filtered_buckets

def test_fft_reduce():
    rate = 44100
    background = np.load("./background.npy")
    test = np.load("./test.npy")
    l = min(test.shape[0], background.shape[0])
    test = test[:l]
    background = background[:l]

    filtered_buckets = fft_subtract(test, background, rate, plot=False)

    np.testing.assert_allclose(background,
        np.fft.ifft(np.fft.fft(background)).real,
        rtol=2
    )
    filtered = np.fft.ifft(filtered_buckets).real.astype(np.int16)
    play_numpy(filtered, rate=RATE)
    play_numpy(test)

# GAN of what is noise vs what is not noise

def text_to_speach():
    import pyttsx3
    engine = pyttsx3.init()
    engine.save_to_file("Hello", "./tmp.wav")
    engine.runAndWait()
    rate, hello = wavfile.read("tmp.wav")
    print(rate, hello)
    play_numpy(hello, rate=rate)
    rate_b, background = wavfile.read(WAVE_OUTPUT_DIR + "background.wav")
    l = min(background.shape[0], hello.shape[0])
    background = background[:l]
    hello = hello[:l]
    combined = hello + background
    play_numpy(combined, rate=rate)
    # Train NN to pull out just hello from combined

def test_play_wav():
    rate_b, background = wavfile.read(WAVE_OUTPUT_DIR + "background.wav")
    rate_t, test = wavfile.read(WAVE_OUTPUT_DIR + "test.wav")
    play_numpy(test, rate=rate_t)
    length_to_modify = min(background.shape[0], test.shape[0])
    print(length_to_modify, test.shape[0], background.shape[0])
    test[:length_to_modify] = test[:length_to_modify] - background[:length_to_modify]
    #play_numpy(test)

if __name__ == "__main__":
    #text_to_speach()
    #test_fft_reduce()
    test_play_wav()
