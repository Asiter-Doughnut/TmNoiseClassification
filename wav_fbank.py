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
    # 打印采样率，单位为赫兹（Hz）
    print('sampling rate: ', rate, 'Hz')
    # 打印位深度
    print('bit depth: ', bit_depth)
    # 打印通道数 和 持续时间，单位为秒（second）
    if len(wav_sample.shape) > 1 and wav_sample.shape[1] > 1:
        print('number of channels: ', wav_sample.shape[1])
        print('duration: ', wav_sample.shape[0] / rate, ' second')
    else:
        print('number of channels: ', 1)
        print('duration: ', wav_sample.shape[0] / rate, ' second')

    # 打印样本数量
    print('number of samples: ', len(wav_sample))
    # 打印类别标签
    print('class: ', class_label)
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

    # 应用梅尔滤波器组到STFT Melspectrum 就是 Fbank
    mel_spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(stft, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()

    # 计算滤波器组能量谱（Filterbank energies）
    # fbank = librosa.feature.filterbank(mel_spectrogram, sr=sr)

    return mel_spectrogram


# 示例用法
# audio_file = './7061-6-0-0.wav'
audio_file = './data/UrbanSound8K/audio/fold3/12647-3-3-0.wav'
# audio_file = './20231114153557.wav'
wav_plotter(audio_file, "test")
fbank_features = extract_fbank_features(audio_file)

print(fbank_features)


# pre-emphasis 预加重
# frame blocking and windowing 分针和加窗
# 分帧的原因是因为 频率在一整个音频里面是变化的 所以我们假设在这小段时间内 频率不变 我们才去做这个傅里叶变化
# Fourier-Transform and Power Spectrum 傅里叶变换和功率谱
# Filter Banks 取得mel_spectrogram
# Mel-frequency Cepstral Coefficients (MFCCs)
# Mean Normalization