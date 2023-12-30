# Loading model
# Processing data
# Processing output
import librosa
import numpy
import scipy
import random

import soundfile

# from rknn.api import RKNN

# wav_path = "../7061-6-0-0.wav"
wav_path = "../20231114153557.wav"
rknn_model = "Class_ptModel.rknn"
loss_weight = "../Class_ptModel_loss.txt"
embedding_weight = "../embedding.txt"


def loading_rockModel_model():
    # Create RKNN object
    global rknn = RKNN(verbose=True)
    # Load RKNN model
    rknn.load_rknn('./resnet_18.rknn')
    rknn.init_runtime()


# wav extraction of mel
def librosa_mel(x):
    mel = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=512, win_length=400, hop_length=160,
                                         window=scipy.signal.windows.hamming, n_mels=80, fmin=20,
                                         fmax=7600)
    return mel


# Read and crop the audio
def processing_audio(audio, num_frames=500):
    if len(audio.shape) >= 2:
        audio = audio[:, 0]
    wab_length = num_frames * 160

    if len(audio) <= wab_length:
        shortage = wab_length - len(audio)
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    # Extract a fixed length of data from an audio randomly.
    start_frame = numpy.int64(random.random() * (len(audio) - wab_length))
    audio = audio[start_frame:start_frame + wab_length - 1]
    audio = numpy.stack([audio], axis=0)
    return audio


def softmax(x):
    return numpy.exp(x) / sum(numpy.exp(x))


def predict(x, show_num=1):
    # get outputs
    # embedding = numpy.loadtxt(embedding_weight) x
    min_val = numpy.min(x)
    max_val = numpy.max(x)
    normalized_arr = (x - min_val) / (max_val - min_val)
    # get loss
    loss = numpy.loadtxt(loss_weight)
    # Get inference
    weighted_sum = numpy.dot(normalized_arr, loss)
    # Conversion result
    label = softmax(weighted_sum)
    top_n_indices = numpy.argpartition(label, -show_num)[-show_num:]
    top_n_values = label[top_n_indices]
    print("Subscript of the first {} maximum value".format(show_num), top_n_indices)
    print("First {} maximum:".format(show_num), top_n_values)


if __name__ == '__main__':
    audio, _ = soundfile.read(wav_path)
    audio = processing_audio(audio)
    audio = librosa_mel(audio)
    outputs = rknn.inference(inputs=[audio])
    predict(outputs)
