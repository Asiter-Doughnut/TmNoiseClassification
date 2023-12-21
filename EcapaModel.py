import sys
import time

import numpy
import numpy as np
import soundfile
import torch.optim

from torch import nn
import torch.nn.functional as F

from loss import AAMsoftmax
from model import ECAPA_TDNN
from util import calculate_eer


class EcapaModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step):
        super(EcapaModel, self).__init__()
        ## ECAPA-TDNN
        self.sound_ecoder = ECAPA_TDNN(C=C).cuda()
        ## Classifier
        self.sound_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        # print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
        #         sum(param.numel() for param in self.sound_ecoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        '''
        train one epoch and return info
        :param epoch: epoch num
        :param loader: dataLoader
        :return: train loss,learn_rate,accuracy
        '''
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(np.array(labels)).cuda()
            sound_embedding = self.sound_ecoder.forward(data.cuda(), aug=False)

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
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, test_list, test_path):
        self.eval()
        files = []
        labels = []
        scores, prlabels = [], []
        # load the test file
        with open(test_path + '/' + test_list, 'r', encoding='utf-8') as file:
            for line in file:
                fileName = line.strip().split('\t')[0]
                label = line.strip().split('\t')[1]
                files.append(fileName)
                labels.append(label)
        for _, (file, label) in enumerate(zip(files, labels)):
            audio, _ = soundfile.read(file)
            # Full sound
            audio = torch.FloatTensor(numpy.stack([audio], axis=0))
            # Process dual-channel audio into single-channel audio.
            if len(audio.shape) >= 3:
                audio = audio[:, :, 0]
            with torch.no_grad():
                embedding = self.sound_ecoder.forward(audio.cuda(), aug=False)
                embedding = F.normalize(embedding, p=2.0, dim=1)
                score, pre_index, = self.predict(embedding)
                print("预测值为%d,得分为%.3f,真实值为%s" % (pre_index, score, label))
                # print(torch.nn.functional.softmax(embedding, dim=-1))
            scores.append(score)
            prlabels.append(0 if label == pre_index else 1)
        EER = calculate_eer(prlabels, scores)
        return EER

    def save_models(self, path):
        '''
        save the models in local
        :return: null
        '''
        torch.save(self.state_dict(), path)

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

    def predict(self, x):
        weighted_sum = torch.matmul(x, self.sound_loss.weight.T)
        label = torch.nn.functional.softmax(weighted_sum, dim=1)
        score, pared = label.topk(1)
        return score, pared
