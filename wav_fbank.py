import librosa
import numpy as np

def extract_fbank_features(audio_file, n_mfcc=13):
    # 加载音频文件
    audio, sr = librosa.load(audio_file)

    # 计算短时傅里叶变换（STFT）
    stft = librosa.stft(audio)

    # 计算梅尔滤波器组（Mel-filterbank）
    mel_filters = librosa.filters.mel(sr, n_fft=2048, n_mfcc=n_mfcc)

    # 应用梅尔滤波器组到STFT
    mel_spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_fft=2048, n_mfcc=n_mfcc)

    # 计算滤波器组能量谱（Filterbank energies）
    fbank = librosa.feature.filterbank(mel_spectrogram, sr=sr, n_mfcc=n_mfcc)

    return fbank


# 示例用法
audio_file = '7061-6-0-0.wav.wav'
fbank_features = extract_fbank_features(audio_file)
print(fbank_features)
