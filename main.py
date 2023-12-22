import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torchaudio

import torch.nn as nn
import argparse
from dataLoader import train_loader
from EcapaModel import EcapaModel
from util import add_arguments, init_dir

parser = argparse.ArgumentParser(description="ECAPA_trainer")
parser = add_arguments(parser)
args = parser.parse_args()
args = init_dir(args)

#  Define the data loader
train_Loader = train_loader(args.test_list, args.path, args.num_frames)
trainLoader = torch.utils.data.DataLoader(train_Loader, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers,
                                          drop_last=True)

s = EcapaModel(lr=args.learning_rate, lr_decay=args.learning_rate_decay, C=args.channel, m=args.amm_m, s=args.amm_s,
               n_class=args.num_class, test_step=args.test_step)

epoch = 1

if __name__ == '__main__':
    s.load_models("./model/ecapa_tdnn_1.model")
    # s.eval_network(test_list=args.train_list, test_path=args.path)
    print(s.eval_network(test_list=args.test_list, test_path=args.path))
    s.load_models("./model/ecapa_tdnn_80.model")
    # s.eval_network(test_list=args.train_list, test_path=args.path)
    print(s.eval_network(test_list=args.test_list, test_path=args.path))
# print(args.batch_size)
# s.save_models()
# loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

# while (1):
#     ## Training for one epoch
#     loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)
#     if epoch >= 80:
#         quit()
#     epoch += 1
