import argparse
import json
import os

import librosa
import numpy
import soundfile
import torch
from flask import Flask, request, jsonify
import torch.nn.functional as F

from EcapaModel import EcapaModel
from util import add_arguments, librosa_mel

app = Flask(__name__)

parser = argparse.ArgumentParser(description="ECAPA_trainer")
parser = add_arguments(parser)
args = parser.parse_args()

# initModel
predictor = EcapaModel(lr=args.learning_rate, lr_decay=args.learning_rate_decay, C=args.channel, m=args.amm_m,
                       s=args.amm_s,
                       n_class=args.num_class, test_step=args.test_step, use_gpu=False)

predictor.load_cpu_location_models('./best_model/ecapa_tdnn_177.pt')

predictor.eval()
print("the model init success!")

labels = {}

score_base = 0

# args.path + '/' + args.label_list
with open(args.path + '/' + 'labels.txt', 'r', encoding='utf-8') as label_file:
    for index, lines in enumerate(label_file):
        labels[index] = lines.replace("\n", '')
    score_base = 1 / len(labels)


@app.route('/api/sound/classification/top', methods=['POST'])
def index():
    file = request.files.get('file')
    label_format = int(request.form.get("labelNumber", 3))

    if file is None:
        return json.dumps({
            'message': "没有音频文件"
        }, ensure_ascii=False)

    audio, _ = soundfile.read(file)
    if len(audio.shape) >= 2:
        audio = audio[:, 0]
    audio = torch.FloatTensor(numpy.stack([librosa_mel(audio)], axis=0))
    embedding = predictor.sound_ecoder.forward(audio, aug=False)
    embedding = F.normalize(embedding, p=2.0, dim=1)
    score, label = predictor.predict(
        embedding, label_format)

    result_array = []

    for score_value, label_value in zip(score, label):
        wav_score = (score_value.item() - score_base) / score_base
        wav_score = 0.8 if (wav_score > 0.8) else wav_score
        result = {
            "label": labels[int(label_value.item())],
            "score": str(round(wav_score * 100, 2)) + "%"
        }
        result_array.append(result)
    # return json.dumps(result_array, )
    # return json.dumps(result_array, ensure_ascii=False)
    return jsonify(result_array)


@app.route('/AudioRecognizeMP3', methods=['POST'])
def audioRecognize():
    try:
        file = request.files.get('file')
        label_format = int(request.form.get("labelNumber", 3))
        audio, _ = soundfile.read(file)
        if len(audio.shape) >= 2:
            audio = audio[:, 0]
        audio = torch.FloatTensor(numpy.stack([librosa_mel(audio)], axis=0))
        embedding = predictor.sound_ecoder.forward(audio, aug=False)
        embedding = F.normalize(embedding, p=2.0, dim=1)
        score, label = predictor.predict(
            embedding, label_format)
        result_array = []
        for score_value, label_value in zip(score, label):
            result = {
                "name": labels[int(label_value.item())],
                "score": round(float(score_value * 100), 3)
            }
            result_array.append(result)
        return jsonify({
            "code": 1000,
            "message": "识别成功",
            "data": result_array,
            "spectrogram ": None
        })
    except Exception as e:
        return jsonify({"code": 1001, "message": "识别失败", "data": [],
                        "spectrogram ": None}), 200


@app.route('/AudioRecognize', methods=['POST'])
def audioRecognizeMP3():
    try:
        file = request.files.get('file')
        label_format = int(request.form.get("labelNumber", 3))
        if file:
            file_path = os.path.join("./", 'main.mp3')
            file.save(file_path)
        # audio, _ = soundfile.read(file)
        audio, _ = librosa.load('./main.mp3', sr=None)
        if len(audio.shape) >= 2:
            audio = audio[:, 0]
        audio = torch.FloatTensor(numpy.stack([librosa_mel(audio)], axis=0))
        embedding = predictor.sound_ecoder.forward(audio, aug=False)
        embedding = F.normalize(embedding, p=2.0, dim=1)
        score, label = predictor.predict(
            embedding, label_format)
        result_array = []
        for score_value, label_value in zip(score, label):
            result = {
                "name": labels[int(label_value.item())],
                "score": round(float(score_value * 100), 3)
            }
            result_array.append(result)
        return json.dumps({
            "code": 1000,
            "message": "识别成功",
            "data": result_array,
            "spectrogram ": None
        }, ensure_ascii=False)
    except Exception as e:
        return jsonify({"code": 1001, "message": "识别失败", "data": [],
                        "spectrogram ": None}), 200
    # return json.dumps({
    #     "code": 1000,
    #     "message": "识别成功",
    #     "data": result_array,
    #     "spectrogram ": None
    # }, ensure_ascii=False)


@app.route('/api/sound/classificationMP3/top', methods=['POST'])
def indexMp3():
    file = request.files.get('file')
    label_format = int(request.form.get("labelNumber", 3))
    webm_path = "./record_audio.webm"
    wav_path = "./record_audio.wav"
    if file is None:
        return json.dumps({
            'message': "没有音频文件"
        }, ensure_ascii=False)
    if file:
        file_path = os.path.join(webm_path)
        file.save(file_path)

    webm_to_wav(webm_path, wav_path, 16000, 1)
    audio, _ = librosa.load(wav_path, sr=None)
    if len(audio.shape) >= 2:
        audio = audio[:, 0]
    audio = torch.FloatTensor(numpy.stack([librosa_mel(audio)], axis=0))
    embedding = predictor.sound_ecoder.forward(audio, aug=False)
    embedding = F.normalize(embedding, p=2.0, dim=1)
    score, label = predictor.predict(
        embedding, label_format)

    result_array = []

    for score_value, label_value in zip(score, label):
        wav_score = (score_value.item() - score_base) / score_base
        wav_score = 0.8 if (wav_score > 0.8) else wav_score
        result = {
            "label": labels[int(label_value.item())],
            "score": str(round(wav_score * 100, 2)) + "%"
        }
        result_array.append(result)
    # return json.dumps(result_array, )
    # return json.dumps(result_array, ensure_ascii=False)
    return jsonify(result_array)


def webm_to_wav(webm_path, wav_path, sampling_rate, channel):
    if os.path.exists(wav_path):
        os.remove(wav_path)
    # command to use ffmpeg change audio/webm to wav
    command = "ffmpeg -loglevel quiet -i {} -ac {} -ar {} {}".format(webm_path, channel, sampling_rate, wav_path)
    os.system(command)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6712, debug=True)
