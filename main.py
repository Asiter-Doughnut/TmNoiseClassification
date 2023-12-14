import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torchaudio

import torch.nn as nn

from dataLoader import train_loader

from EcapaModel import EcapaModel

# audio, sr = soundfile.read('20231114153557.wav')


# test_list
train_Loader = train_loader('test_list.txt', './data', 300)
trainLoader = torch.utils.data.DataLoader(train_Loader, batch_size=64, shuffle=True, num_workers=0,
                                          drop_last=True)

epoch = 1
# lr, lr_decay, C, n_class, m, s, test_step
s = EcapaModel(lr=0.001, lr_decay=0.97, C=512, m=0.2, s=30, n_class=10, test_step=1)
#

if __name__ == '__main__':
    while (1):
        ## Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)
        if epoch >= 80:
            quit()
        epoch += 1
