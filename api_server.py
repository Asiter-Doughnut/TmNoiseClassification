import argparse
import json
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6712, debug=True)
