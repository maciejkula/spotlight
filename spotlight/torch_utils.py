import numpy as np

import torch


def _slice_or_none(arg, slc):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return tuple(x[slc] for x in arg)
    else:
        return arg[slc]


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


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)
    length = min(len(x) for x in tensors if hasattr(x, '__len__'))

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, length, batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, length, batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def grouped_minibatch(groupby_array, *tensors):

    values, group_indices = np.unique(groupby_array, return_index=True)
    group_indices = np.concatenate((group_indices, [len(groupby_array)]))

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(len(values)):
            slc = slice(group_indices[i], group_indices[i + 1])
            yield tensor[slc]
    else:
        for i in range(len(values)):
            slc = slice(group_indices[i], group_indices[i + 1])
            yield tuple(x[slc] for x in tensors)


def shuffle(*arrays, **kwargs):

    random_state = kwargs.get('random_state')

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def set_seed(seed, cuda=False):

    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)


def concatenate(tensors, axis=0):

    inputs = []

    for tensor in tensors:
        if tensor is None:
            continue
        elif isinstance(tensor, tuple()):
            inputs.extend(tensor)
        else:
            inputs.append(tensor)

    return torch.cat(inputs, dim=axis)
