# 这是一个示例 Python 脚本。
import librosa
import numpy
import torch
import torchaudio
from matplotlib import pyplot as plt

from dataLoader import train_loader
from model import ECAPA_TDNN

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128


def getTorchMel(x, sample_rate):
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=512, win_length=400, hop_length=160,
    #                                                        f_min=20, f_max=7600, window_fn=torch.hamming_window,
    #                                                        n_mels=80)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )
    return mel_spectrogram(x)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    print(librosa.power_to_db(specgram))
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.show()


net = ECAPA_TDNN(1024)

print(train_loader('train_list.txt', './data').getList())

if __name__ == '__main__':
    warAudio, sr = librosa.load('./7061-6-0-0.wav')
    # melspec = getTorchMel(torch.from_numpy(warAudio), sr)
    # print(melspec.shape)
    # plot_spectrogram(melspec, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    # warAudio, sr = librosa.load('./20231114153557.wav')
    # melspec = getTorchMel(torch.from_numpy(warAudio), sr)
    # print(melspec.shape)
    # plot_spectrogram(melspec, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    # print(torch.from_numpy(warAudio).shape)
    # net.forward(torch.from_numpy(warAudio), aug=False)
    # warAudio = numpy.stack(warAudio, axis=0)
    # print(torch.tensor(warAudio))
