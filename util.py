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


def add_arguments():
    with open('./config/ecapaModel.yml', encoding='utf-8') as file:
        content = file.read()
        print(content)
        print(type(content))

        print("\n*****转换yaml数据为字典或列表*****")
        # 设置Loader=yaml.FullLoader忽略YAMLLoadWarning警告
        data = yaml.load(content, Loader=yaml.FullLoader)
        print(data)
        print(type(data))
        print(data.get('my'))  # 类型为字典 <class 'dict'>
