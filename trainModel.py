import glob
import os

import torch

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

# find the model in the directory
modelfiles = glob.glob('%s/ecapa_tdnn_*.model' % args.model_save_path)
modelfiles.sort()

# if model is exit,continue train the previous model
if len(modelfiles) >= 1:
    print("model %s have trained record" % modelfiles[-1])
    filename, _ = os.path.splitext(os.path.basename(modelfiles[-1]))
    epoch = int(filename[11:]) + 1
    s = EcapaModel(lr=args.learning_rate, lr_decay=args.learning_rate_decay, C=args.channel, m=args.amm_m, s=args.amm_s,
                   n_class=args.num_class, test_step=args.test_step)
    s.load_models(modelfiles[-1])
# no model,init the model
else:
    epoch = 1
    s = EcapaModel(lr=args.learning_rate, lr_decay=args.learning_rate_decay, C=args.channel, m=args.amm_m, s=args.amm_s,
                   n_class=args.num_class, test_step=args.test_step)

while 1:
    # trian one epoch
    loss, lr, acc = s.train_network(epoch=epoch, loader=train_loader)

    # record the epoch step train
    if epoch % args.test_step == 0:
        s.save_models(args.model_save_path + 'ecapa_tdnn_%s.model' % epoch)

    # epoch more than quit train
    if epoch >= args.max_epoch:
        quit()

    epoch *= 1
