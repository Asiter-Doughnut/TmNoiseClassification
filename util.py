import os.path
import yaml


def accuracy(output, target, turek=(1,)):
    """
    :param output: Model output
    :param target: True label
    :param turek: K value tuple
    :return: Accuracy array
    """
    maxk = max(turek)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in turek:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def init_dir(args):
    '''
    initialize a folder to dir
    '''
    args.model_save_path = os.path.join(args.save_path, 'model')
    args.record_save_path = os.path.join(args.save_path, 'model_record.txt')
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def add_arguments(parser):
    with open('./config/ecapaModel.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            if isinstance(value, dict):
                add_dist_arguments(parser, value)
            else:
                parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser


def add_dist_arguments(parser, dist_list):
    for key, value in dist_list.items():
        if isinstance(value, dict):
            add_dist_arguments(parser, value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)


def extract_number(fileName,save_path):
    return int(fileName.replace(".model", '')[len(save_path) + 12:])
