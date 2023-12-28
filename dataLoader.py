import random

import librosa
import numpy
import numpy as np
import scipy.signal.windows
import soundfile
import torch
import torchaudio


class train_loader(object):
    def __init__(self, train_list, train_path, num_frames, **kwargs):
        # Load data & labels
        self.num_frames = num_frames
        self.data_list = []
        self.data_label = []
        with open(train_path + '/' + train_list, 'r', encoding='utf-8') as file:
            for line in file:
                label = line.strip().split('\t')[1]
                fileName = line.strip().split('\t')[0]
                self.data_list.append(fileName)
                self.data_label.append(label)

    def __getitem__(self, index):
        # In the later stage, noise will be added to increase the accuracy of training.
        # Read the wavfile and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        # Testing Mono Audio
        # audio, sr = soundfile.read("20231114153557.wav")
        # compute the wav length
        # + 240
        length = self.num_frames * 160 + 240
        # If the audio length is shorter than the required wav length, we just need to join the wav.
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        # Extract a fixed length of data from an audio randomly.
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Process dual-channel audio into single-channel audio.
        if len(audio.shape) >= 3:
            audio = audio[:, :, 0]
        print("self.fbank", self.fbank(torch.FloatTensor(audio[0])))
        print("self.fbank2", torch.Tensor(self.fbank2(audio[0])))
        return torch.Tensor(self.fbank2(audio[0])), int(self.data_label[index])

    def fbank(self, x):
        torchfbank = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                          f_min=20, f_max=7600, window_fn=torch.hamming_window,
                                                          n_mels=80)
        x = torchfbank(x)
        return x

    def fbank2(self, x):
        stft = librosa.stft(x, n_fft=512, hop_length=160, win_length=400, window=scipy.signal.windows.hamming)
        stft_normalized = librosa.util.normalize(np.abs(stft))
        mfcc = librosa.feature.mfcc(S=stft_normalized, n_mfcc=80)
        return mfcc

    def __len__(self):
        return len(self.data_list)
