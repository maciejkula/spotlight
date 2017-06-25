def gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def minibatch(tensor, batch_size):

    for i in range(0, len(tensor), batch_size):
        yield tensor[i:i + batch_size]
