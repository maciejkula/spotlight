import numpy as np

import torch
from torch.autograd import Variable

from spotlight.torch_utils import gpu


def _fnc_or_none(arg, fnc):

    if arg is None:
        return None
    if isinstance(arg, tuple):
        return tuple(fnc(x) for x in arg)
    else:
        return fnc(arg)


def _prepare(tensor, required_rows, use_cuda):

    required_size = (required_rows,) + tensor.shape[1:]

    tensor = torch.from_numpy(tensor).expand(required_size)

    return Variable(gpu(tensor, use_cuda))


def _predict_process_ids(user_ids, item_ids, num_items, use_cuda):

    if item_ids is None:
        item_ids = np.arange(num_items, dtype=np.int64)

    if np.isscalar(user_ids):
        user_ids = np.array(user_ids, dtype=np.int64)

    user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
    item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

    if item_ids.size()[0] != user_ids.size(0):
        user_ids = user_ids.expand(item_ids.size())

    user_var = Variable(gpu(user_ids, use_cuda))
    item_var = Variable(gpu(item_ids, use_cuda))

    return user_var, item_var


def _predict_process_features(user_features, context_features, item_features,
                              required_rows,
                              use_cuda):

    fnc = lambda x: _prepare(x, required_rows, use_cuda)

    user_features = _fnc_or_none(user_features, fnc)
    context_features = _fnc_or_none(context_features, fnc)
    item_features = _fnc_or_none(item_features, fnc)

    return user_features, context_features, item_features
