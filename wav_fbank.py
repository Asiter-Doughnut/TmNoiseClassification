import librosa
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
import struct


# 绘图wav函数
def wav_plotter(full_path, class_label):
    rate, wav_sample = wav.read(full_path)
    wave_file = open(full_path, "rb")
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H", bit_depth_string)[0]
    print('sampling rate: ', rate, 'Hz')  # 打印采样率，单位为赫兹（Hz）
    print('bit depth: ', bit_depth)  # 打印位深度
    print('number of channels: ', wav_sample.shape[1])  # 打印通道数
    print('duration: ', wav_sample.shape[0] / rate, ' second')  # 打印持续时间，单位为秒（second）
    print('number of samples: ', len(wav_sample))  # 打印样本数量
    print('class: ', class_label)  # 打印类别标签
    plt.figure(figsize=(12, 4))
    plt.plot(wav_sample)
    plt.show()


def extract_fbank_features(audio_file, n_mfcc=13):
    # 加载音频文件
    audio, sr = librosa.load(audio_file)

    # 计算短时傅里叶变换（STFT）
    stft = librosa.stft(audio)

    # 计算梅尔滤波器组（Mel-filterbank）
    mel_filters = librosa.filters.mel(sr=sr, n_fft=2048)

    # 应用梅尔滤波器组到STFT
    mel_spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr)

    # 计算滤波器组能量谱（Filterbank energies）
    # fbank = librosa.feature.filterbank(mel_spectrogram, sr=sr)

    return mel_spectrogram


# 示例用法
audio_file = './7061-6-0-0.wav'
fbank_features = extract_fbank_features(audio_file)
print(fbank_features)
