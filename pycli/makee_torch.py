import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import torch

import scipy.io.wavfile


CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000 #44100

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class SinLayer(torch.nn.Module):
    def __init__(self, sin_part_sizes) -> None:
        super().__init__()

        self.sin_part_sizes = sin_part_sizes
        size = (sum(sin_part_sizes), )

        self.amps = torch.rand(size=(len(sin_part_sizes),))
        self.amps = torch.nn.Parameter(self.amps)
        self.amp_mult = torch.zeros(size=size)

        self.freqs = torch.rand(size=(len(sin_part_sizes),))
        self.freqs = torch.nn.Parameter(self.freqs)
        self.freqs_mult = torch.zeros(size=size)

        self.freq_offsets = torch.rand(size=(len(sin_part_sizes),))
        self.freq_offsets = torch.nn.Parameter(self.freq_offsets)
        self.freq_offsets_mult = torch.zeros(size=size)

        self.shifts = torch.rand(size=(len(sin_part_sizes),))
        self.shifts = torch.nn.Parameter(self.shifts)
        self.shifts_mult = torch.zeros(size=size)

        self.first_val = torch.rand(1)
        self.first_val = torch.nn.Parameter(self.first_val)

        begin = 0
        for i in range(len(sin_part_sizes)):
            self.amp_mult[begin:begin+sin_part_sizes[i]] = self.amps[i]
            self.freqs_mult[begin:begin+sin_part_sizes[i]] = self.freqs[i]
            self.freq_offsets_mult[begin:begin+sin_part_sizes[i]] = self.freq_offsets[i]
            self.shifts_mult[begin:begin+sin_part_sizes[i]] = self.shifts[i]
            begin += sin_part_sizes[i]

        self.amp_mult = torch.nn.Parameter(self.amp_mult)
        self.freqs_mult = torch.nn.Parameter(self.freqs_mult)
        self.freq_offsets_mult = torch.nn.Parameter(self.freq_offsets_mult)
        self.shifts_mult = torch.nn.Parameter(self.shifts_mult)


    def forward(self, x):
        amp = self.amp_mult
        freq = self.freqs_mult
        freq_offset = self.freq_offsets_mult
        shift = self.shifts_mult
        y = amp * torch.sin(freq * x + freq_offset) + shift
        y[0] = self.first_val
        return y

        y = torch.zeros(size=x.shape, device=device)
        begin = 0

        for i in range(len(self.sin_part_sizes)):
            xi = x[begin:begin+self.sin_part_sizes[i]]

            amp = self.amps[i]
            freq = self.freqs[i]
            shift = self.shifts[i]
            freq_offset = self.freq_offsets[i]
            yi = amp * torch.sin(freq * xi + freq_offset) + shift
            y[begin:begin+self.sin_part_sizes[i]] = yi

            begin += self.sin_part_sizes[i]
        y[0] = self.first_val
        return y

class SinNN(torch.nn.Module):
    def __init__(self, sin_part_sizes, num_sections) -> None:
        super(SinNN, self).__init__()
        self.sin_part_sizes = sin_part_sizes
        self.layers = torch.nn.ModuleList(
            [SinLayer(sin_part_sizes) for _ in range(num_sections)]
        )

    def forward(self, X):
        Y = torch.zeros(size=X.shape, device=device)
        for i in range(X.shape[0]):
            x = X[i]
            y = self.layers[i](x)
            Y[i] = y
        return Y

def main():
    file_name = "./data/recordings/test3.wav"
    input_audio, _ = librosa.load(file_name, sr=16000)
    input_audio = torch.tensor(input_audio)

    start = 6302
    end = start + 3200
    audio_sample = input_audio[start:end]
    section_lengths = [
        78, 78, 79, 80, 80, 82, 83, 84, 84, 86,
        87, 91, 91, 95, 99, 105, 112
    ]

    input_size = max(section_lengths)
    X = torch.zeros(size=(len(section_lengths), input_size))
    Y = torch.zeros(size=(len(section_lengths), input_size))

    sin_part_idxes = torch.tensor([
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
    #sin_part_idxes = torch.concat(
    #    [ torch.arange(0,112, 5), torch.tensor([112]) ]
    #)
    sin_part_sizes = []
    for i in range(1, len(sin_part_idxes)):
        sin_part_sizes.append(sin_part_idxes[i] - sin_part_idxes[i-1])

    begin = 0
    for i in range(len(section_lengths)):
        x = torch.arange(0, input_size)
        X[i] = x

        y = torch.zeros(size=(input_size,))
        y[:section_lengths[i]] = audio_sample[begin:begin+section_lengths[i]]
        Y[i] = y

        begin += section_lengths[i]

    X = X.to(device)
    Y = Y.to(device)
    print(X, Y)

    lr = 0.1
    model = SinNN(sin_part_sizes=sin_part_sizes, num_sections=X.shape[0]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_model(model, optimizer, X, Y)
    eval_model(model, X, Y, section_lengths, play_audio=True)

def eval_model(model, X, Y, section_sizes, play_audio=False):
    y_hat = model(X)
    output_audio = []
    actual_audio = []
    for i in range(X.shape[0]):
        plt.subplot(X.shape[0], 1, i+1)
        plt.plot(Y[i].detach().numpy())
        plt.plot(y_hat[i].detach().numpy())

        actual_audio.append(Y[i, 0:section_sizes[i]].detach().numpy())
        output_audio.append(y_hat[i, 0:section_sizes[i]].detach().numpy())

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

        scipy.io.wavfile.write(
            "./data/recordings/makee_actual_manual_idx.wav", RATE,
            np.tile(actual_audio, num_repeats)
        )
        scipy.io.wavfile.write(
            "./data/recordings/makee_generated_manual_idx.wav", RATE,
            np.tile(output_audio, num_repeats)
        )

def train_model(model, optimizer, X, Y):
    loss_fn = torch.nn.MSELoss()
    model.train()

    time_in_forward = 0.0
    time_in_backward = 0.0
    for train_itr in range(40001):
        t0 = time.time()
        y_hat = model(X)
        loss = loss_fn(y_hat, Y)
        t1 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        time_in_forward += t1 - t0
        time_in_backward += t2 - t1
        if train_itr % 100 == 0:
            print("Train:", train_itr, " error:", loss,
                "forward:", time_in_forward,
                "backward:", time_in_backward)

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