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
