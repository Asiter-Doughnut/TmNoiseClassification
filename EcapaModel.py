import sys
import time

import librosa
import numpy
import numpy as np
import scipy
import soundfile
import torch.optim
import torchaudio
import tqdm

from torch import nn
import torch.nn.functional as F

from loss import AAMsoftmax
from model import ECAPA_TDNN
from util import calculate_eer, calculate_min_dcf, librosa_mel


class EcapaModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, use_gpu):
        super(EcapaModel, self).__init__()
        self.sound_ecoder = ECAPA_TDNN(C=C)
        self.sound_loss = AAMsoftmax(n_class=n_class, m=m, s=s)
        self.use_gpu = use_gpu
        if use_gpu:
            self.sound_ecoder = self.sound_ecoder.cuda()
            self.sound_loss = self.sound_loss.cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.sound_ecoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        '''
        train one epoch and return info
        :param epoch: epoch num
        :param loader: dataLoader
        :return: train loss,learn_rate,accuracy
        '''
        self.train()
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(np.array(labels))
            if self.use_gpu:
                data = data.cuda()
                labels = labels.cuda()
            sound_embedding = self.sound_ecoder.forward(data, aug=True)
            nloss, prec = self.sound_loss.forward(sound_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        # epoch - 1
        self.scheduler.step()
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, test_list, test_path):
        self.eval()
        files = []
        labels = []
        scores, predict_labels = [], []

        with open(test_path + '/' + test_list, 'r', encoding='utf-8') as file:
            for line in file:
                fileName = line.strip().split('\t')[0]
                label = line.strip().split('\t')[1]
                files.append(fileName)
                labels.append(label)

        for _, (file, label) in tqdm.tqdm(enumerate(zip(files, labels)), total=len(files)):
            audio, _ = soundfile.read(file)

            if len(audio.shape) >= 2:
                audio = audio[:, 0]

            # torch mel
            # audio = numpy.stack([audio], axis=0)
            # audio = torch.FloatTensor(audio)
            # audio = self.fbank(audio)

            # librosa mel
            audio = torch.FloatTensor(numpy.stack([librosa_mel(audio)], axis=0))

            with torch.no_grad():
                if self.use_gpu:
                    audio = audio.cuda()
                embedding = self.sound_ecoder.forward(audio, aug=False)
                embedding = F.normalize(embedding, p=2.0, dim=1)
                score, pre_index, = self.predict(embedding, 1)

            scores.append(score[0].cpu().numpy())
            predict_labels.append(1 if int(label) == pre_index else 0)
        EER, fpr, tpr = calculate_eer(predict_labels, scores)
        min_DCF = calculate_min_dcf(fpr, tpr)
        return EER, min_DCF

    def fbank(self, x):
        torchfbank = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                          f_min=20, f_max=7600, window_fn=torch.hamming_window,
                                                          n_mels=80)
        x = torchfbank(x)
        return x

    def fbank2(self, x):
        melspectrogram = librosa.feature.melspectrogram(y=x, sr=1600, n_fft=512, win_length=400, hop_length=160,
                                                        window=scipy.signal.windows.hamming, n_mels=80, fmin=20,
                                                        fmax=7600)
        return melspectrogram

    def save_models(self, path):
        '''
        save the models in local
        :return: null
        '''
        torch.save(self.state_dict(), path)

    def save_jit_trace_models(self):
        # modelInput soundLength*160
        example_forward_input = torch.rand([1, 80, 500 * 160])
        # set model be eval patten
        self.sound_ecoder.eval()
        traced_model = torch.jit.trace(self.sound_ecoder.forward, example_forward_input)
        # save model
        torch.jit.save(traced_model, "Class_ptModel.pt")
        tensor = self.sound_loss.weight.T
        numpy_array = tensor.cpu().detach().numpy()
        # save loss weight
        np.savetxt('Class_ptModel_loss.txt', numpy_array)

    def load_models(self, path):
        '''
        load the models
        :return:null
        '''
        self_state = self.state_dict()
        loader_state = torch.load(path)
        for name, param in loader_state.items():
            if name not in self_state:
                print("%s is not in the model." % name)
                continue
            if loader_state[name].size() != self_state[name].size():
                print("Wrong parameter length:%s ,model:%s,loadedModel:%s" % (
                    name, self_state[name].size(), loader_state[name].size()))
                continue
            self_state[name].copy_(param)

    def predict(self, x, show_num):
        weighted_sum = torch.matmul(x, self.sound_loss.weight.T)
        label = torch.nn.functional.softmax(weighted_sum, dim=1)[0]
        score, pared = label.topk(show_num)
        return score, pared
