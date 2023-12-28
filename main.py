import argparse
from EcapaModel import EcapaModel
from dataLoader import train_loader
from util import add_arguments, init_dir

parser = argparse.ArgumentParser(description="ECAPA_trainer")
parser = add_arguments(parser)
args = parser.parse_args()
args = init_dir(args)

s = EcapaModel(lr=args.learning_rate, lr_decay=args.learning_rate_decay, C=args.channel, m=args.amm_m, s=args.amm_s,
               n_class=args.num_class, test_step=args.test_step, useGPU=False)


parser = argparse.ArgumentParser(description="ECAPA_trainer")
parser = add_arguments(parser)
args = parser.parse_args()
args = init_dir(args)
train_Loader = train_loader(args.train_list, args.path, args.num_frames)

if __name__ == '__main__':
    # train_Loader.__getitem__(2)
    score = s.eval_network(test_list=args.test_list, test_path=args.path)
    # s.load_models("./model/ecapa_tdnn_178.pt")
    # s.save_jit_trace_models()
