

import importlib
import time

import librosa

import wav2vec_test

def main():
    file_name = "./data/recordings/test3.wav"
    input_audio, _ = librosa.load(file_name, sr=16000)
    while True:
        time.sleep(1)
        importlib.reload(wav2vec_test)
        wav2vec_test.make_e_of_test_reload(input_audio)

if __name__ == "__main__":
    main()