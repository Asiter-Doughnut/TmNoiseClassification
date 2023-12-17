import sys
import time

import numpy as np
import torch.optim
from torch import nn

from loss import AAMsoftmax
from model import ECAPA_TDNN


class EcapaModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step):
        super(EcapaModel, self).__init__()
        ## ECAPA-TDNN
        self.sound_ecoder = ECAPA_TDNN(C=C).cuda()
        ## Classifier
        self.sound_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.sound_ecoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
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

    def save_models(self):
        '''
        save the models in local
        :return: null
        '''
        torch.save(self.state_dict(), './ecapa_tdnn/tensor.model')

    def load_models(self):
        '''
        load the models
        :return:null
        '''
        self_state = self.state_dict()
        loader_state = torch.load('./ecapa_tdnn/tensor.model')
        for name, param in loader_state.items():
            if name not in self_state:
                print("%s is not in the model." % name)
                continue
            if loader_state[name].size() != self_state[name].size():
                print("Wrong parameter length:%s ,model:%s,loadedModel:%s" % (
                    name, self_state[name].size(), loader_state[name].size()))
                continue
            self_state[name].copy_(param)
