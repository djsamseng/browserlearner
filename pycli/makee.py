
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000 #44100

class SinModel():
    def __init__(self, sin_part_sizes) -> None:
        self.sin_part_sizes = sin_part_sizes
        self.amps = np.random.random(size=(len(sin_part_sizes),))
        self.freqs = np.random.random(size=(len(sin_part_sizes),))
        self.freq_offsets = np.random.random(size=(len(sin_part_sizes),))
        self.shifts = np.random.random(size=(len(sin_part_sizes),))
        self.first_val = np.random.random(1)

    def copy(self):
        model = SinModel(self.sin_part_sizes)
        model.amps = self.amps.copy()
        model.freqs = self.freqs.copy()
        model.shifts = self.shifts.copy()
        model.first_val = self.first_val
        return model

    def forward(self, x):
        y = np.zeros(shape=x.shape)
        begin = 0
        for i in range(len(self.sin_part_sizes)):
            xi = x[begin:begin+self.sin_part_sizes[i]]
            amp = self.amps[i]
            freq = self.freqs[i]
            shift = self.shifts[i]
            freq_offset = self.freq_offsets[i]
            #freq_offset = 0.0
            #if begin == 39 or begin == 63 or begin == 70:
            #    freq_offset = -np.pi/2
            #if begin == 50:
            #    freq_offset = np.pi/2
            yi = amp * np.sin(freq * xi + freq_offset) + shift
            y[begin:begin+self.sin_part_sizes[i]] = yi

            begin += self.sin_part_sizes[i]
        y[0] = self.first_val
        return y

    def backward(self, x, error_grad_arr, lr):
        begin = 0
        for i in range(len(self.sin_part_sizes)):
            self.backward_i(x, error_grad_arr, lr, i, begin)
            begin += self.sin_part_sizes[i]

        self.first_val -= error_grad_arr[0] * lr

    def backward_i(self, x, error_grad_arr, lr, i ,begin):
        xi = x[begin:begin+self.sin_part_sizes[i]]
        amp = self.amps[i]
        freq = self.freqs[i]
        shift = self.shifts[i]
        error_grad = error_grad_arr[begin:begin+self.sin_part_sizes[i]]

        freq_offset = self.freq_offsets[i]
        #freq_offset = 0.0
        #if begin == 39 or begin == 63 or begin == 70:
        #    freq_offset = -np.pi/2
        #if begin == 50:
        #    freq_offset = np.pi/2

        d_amp = error_grad * (np.sin(freq * xi + freq_offset))
        d_freq = error_grad * (amp * np.cos(freq * xi + freq_offset) * xi)
        d_shift = error_grad * (1.0)
        d_freq_offset = error_grad * (amp * np.cos(freq * xi + freq_offset))

        self.amps[i] -= np.sum(d_amp * lr)
        self.freqs[i] -= np.sum(d_freq * lr)
        self.shifts[i] -= np.sum(d_shift * lr)
        self.freq_offsets -= np.sum(d_freq_offset * lr)


    def print(self):
        begin = 0
        for i in range(len(self.sin_part_sizes)):
            amp = self.amps[i]
            freq = self.freqs[i]
            shift = self.shifts[i]
            print(
                "=====",
                str(begin),":", str(begin+self.sin_part_sizes[i]),
                "====="
            )
            print("amp:", amp)
            print("freq:", freq)
            print("shift:", shift)

def main():
    file_name = "./data/recordings/test3.wav"
    input_audio, _ = librosa.load(file_name, sr=16000)

    start = 6302
    end = start + 3200
    audio_sample = input_audio[start:end]
    section_lengths = [
        78, 78, 79, 80, 80, 82, 83, 84, 84, 86,
        87, 91, 91, 95, 99, 105, 112
    ]

    input_size = max(section_lengths)
    X = np.zeros(shape=(len(section_lengths), input_size))
    Y = np.zeros(shape=(len(section_lengths), input_size))

    sin_part_idxes = np.array([
        0,
        7,
        14,
        21,
        27,
        35,
        39,
        50,
        63,
        70,
        80,
        92,
        100,
        104,
        112
    ])
    sin_part_idxes = np.concatenate(([np.arange(0,112, 5), [112]]))
    sin_part_sizes = []
    for i in range(1, len(sin_part_idxes)):
        sin_part_sizes.append(sin_part_idxes[i] - sin_part_idxes[i-1])

    begin = 0
    for i in range(len(section_lengths)):
        x = np.arange(0, input_size)
        '''
        x[7:] = np.arange(0, input_size-7)
        x[27:] = np.arange(0, input_size-27)
        x[35:] = np.arange(0, input_size-35)
        x[39:] = np.arange(0, input_size-39)
        x[50:] = np.arange(0, input_size-50)
        x[63:] = np.arange(0, input_size-63)
        x[70:] = np.arange(0, input_size-70)
        x[80:] = np.arange(0, input_size-80)
        x[92:] = np.arange(0, input_size-92)
        x[100:] = np.arange(0, input_size-100)
        x[104:] = np.arange(0, input_size-104)
        '''
        X[i] = x

        y = np.zeros(shape=(input_size,))
        y[:section_lengths[i]] = audio_sample[begin:begin+section_lengths[i]]
        Y[i] = y

        begin += section_lengths[i]

    print(X, Y)
    models = []
    for _ in range(X.shape[0]):
        model = SinModel(sin_part_sizes=sin_part_sizes)
        models.append(model)

    def get_error_by_section(error, sin_part_idxes, sin_part_sizes):
        error_by_section = []
        for j in range(len(sin_part_sizes)):
            error_section = error[
                sin_part_idxes[j]:sin_part_idxes[j]+sin_part_sizes[j]
            ]
            error_by_section.append(np.sum(error_section))
        return np.array(error_by_section)

    lr = 1.0
    for train_itr in range(5001):
        if train_itr / 100 % 4 == 0.0:
            # Every 400
            for i in range(X.shape[0]):
                x = X[i]
                y = Y[i]

                model = models[i]

                model_copy = model.copy()
                model_copy.freqs *= 2

                for _ in range(100):
                    y_hat_model_copy = model_copy.forward(x)
                    error_grad_model_copy = 2. * (y_hat_model_copy - y) / len(y)
                    model_copy.backward(
                        x=x, error_grad_arr=error_grad_model_copy, lr=lr)

                y_hat = model.forward(x)
                error = (y_hat - y) ** 2
                error_by_section = get_error_by_section(
                    error,
                    sin_part_idxes=sin_part_idxes,
                    sin_part_sizes=sin_part_sizes
                )

                y_hat_model_copy = model_copy.forward(x)
                error_model_copy = (y_hat_model_copy - y) ** 2
                error_by_section_model_copy = get_error_by_section(
                    error_model_copy,
                    sin_part_idxes=sin_part_idxes,
                    sin_part_sizes=sin_part_sizes
                )
                freqs_to_double = np.where(error_by_section_model_copy < error_by_section)[0]
                print(
                    "Doubling freqs:", freqs_to_double,
                )
                model.freqs[freqs_to_double] *= 2


        for i in range(X.shape[0]):
            # TODO: shuffle frequencies (double/half)
            # It's very easy to get caught at a frequency that is double/half
            # the desired since a small change is upward in the gradient slope
            x = X[i]
            y = Y[i]
            model = models[i]
            y_hat = model.forward(x)

            error = np.mean((y_hat - y) ** 2)
            if train_itr % 100 == 0:
                print("Train:", train_itr, "Model:", i, " error:", error)
            error_grad_arr = 2. * (y_hat - y) / len(y)
            model.backward(x=x, error_grad_arr=error_grad_arr, lr=lr)

    def evaluate_model(X, Y, models, section_sizes, play_audio=False):
        output_audio = []
        actual_audio = []
        begin = 0
        for i in range(X.shape[0]):
            x = X[i]
            y = Y[i]
            model = models[i]
            y_hat = model.forward(x)
            plt.subplot(X.shape[0], 1, i+1)
            plt.plot(y)
            plt.plot(y_hat)

            actual_audio.append(y[0:section_sizes[i]])
            output_audio.append(y_hat[0:section_sizes[i]])

        actual_audio = np.concatenate(actual_audio).astype(np.float32)
        output_audio = np.concatenate(output_audio).astype(np.float32)

        plt.show()

        if play_audio:
            num_repeats = 20
            print("===== Actual audio =====")
            play_numpy(np.tile(actual_audio, num_repeats))
            time.sleep(1)
            print("===== Output audio =====")
            play_numpy(np.tile(output_audio, num_repeats))


    evaluate_model(X, Y, models, section_lengths, play_audio=True)

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

if __name__ == "__main__":
    main()