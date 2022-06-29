import logging
import time
from pathlib import Path

from scipy.special import softmax
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} > {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def make_same_size(logits, target):
    assert isinstance(logits, torch.Tensor), "model output {}".format(type(logits))
    size = logits.size()
    if logits.size() != target.size():
        if len(size) == 5:
            # mark : change align_corners to True
            logits = F.interpolate(logits, target.size()[2:], align_corners=True, mode='trilinear')
        elif len(size) == 4:
            logits = F.interpolate(logits, target.size()[2:], align_corners=True, mode='bilinear')
        else:
            raise Exception("Invalid size of logits : {}".format(size))
    return logits


def to_cuda(t, device=None):
    if isinstance(t, (torch.nn.Module, torch.Tensor)):
        return t.cuda(device)
    elif isinstance(t, (list, tuple)):
        l = []
        for i in t:
            l.append(to_cuda(i))
        return l
    elif isinstance(t, torch.optim.Optimizer):
        for state in t.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(device)
        return t
    elif hasattr(t, 'cuda'):
        return t.cuda(device)
    else:
        return t


def to_numpy(tensor):
    if isinstance(tensor, (tuple, list)):
        res = []
        for t in tensor:
            res.append(to_numpy(t))
        return res
    else:
        # <class 'numpy.int64'>
        if isinstance(tensor, np.ndarray) or str(type(tensor))[8:13] == 'numpy':
            return tensor
        else:
            return tensor.detach().cpu().numpy()


def get_prediction(logits, cpu=False):
    size = logits.shape
    if cpu:
        softmax_func = softmax
        argmax_func = np.argmax
        logits = to_numpy(logits)
    else:
        softmax_func = F.softmax
        argmax_func = torch.argmax

    if size[1] > 1:
        preds = argmax_func(softmax_func(logits, 1), 1)
    else:
        if cpu:
            raise Exception("Only support gpu implementation.")
        size = list(size)
        # delete channel dim to prevent
        size = [size[0]] + size[2:]
        preds = torch.round(torch.sigmoid(logits)).long().reshape(size)
    return preds


class Timer(object):
    def __init__(self, verbose=False):
        self.start_time = time.time()
        self.verbose = verbose
        self.duration = 0

    def restart(self):
        self.duration = self.start_time = time.time()
        return self.duration

    def stop(self):
        time.asctime()
        return time.time() - self.start_time

    def get_last_duration(self):
        return self.duration

    def sec2time(self, seconds):
        return seconds / 3600, seconds / 60, seconds % 60

    def get_formatted_duration(self, duration=None):
        if duration is None:
            duration = self.duration
        return 'Time {:^.4f} h, {:^.4f} m, {:^.4f} s'.format(*self.sec2time(duration))

    def __enter__(self):
        self.restart()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = self.stop()
        if self.verbose:
            print(self.get_formatted_duration())


def mkdir(path, level=2, create_self=True):
    """ Make directory for this path,
    level is how many parent folders should be created.
    create_self is whether create path(if it is a file, it should not be created)
    e.g. : mkdir('/home/parent1/parent2/folder', level=3, create_self=False),
    it will first create parent1, then parent2, then folder.
    :param path: string
    :param level: int
    :param create_self: True or False
    :return:
    """
    p = Path(path)
    if create_self:
        paths = [p]
    else:
        paths = []
    level -= 1
    while level != 0:
        p = p.parent
        paths.append(p)
        level -= 1

    for p in paths[::-1]:
        p.mkdir(exist_ok=True)


def put_theta(model, theta):
    def k_param_fn(tmp_model, name=None):
        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + '.' + k))
        else:
            for (k, v) in tmp_model._parameters.items():
                if not isinstance(v, torch.Tensor):
                    continue
                tmp_model._parameters[k] = theta[str(name + '.' + k)]

    k_param_fn(model)
    return model


def get_updated_network(old, new, lr, load=False):
    updated_theta = {}
    state_dicts = old.state_dict()
    param_dicts = dict(old.named_parameters())

    for i, (k, v) in enumerate(state_dicts.items()):
        if k in param_dicts.keys() and param_dicts[k].grad is not None:
            updated_theta[k] = param_dicts[k] - lr * param_dicts[k].grad
        else:
            updated_theta[k] = state_dicts[k]
    if load:
        new.load_state_dict(updated_theta)
    else:
        new = put_theta(new, updated_theta)
    return new


def get_logger(logger_name, filename, file_mode='a', terminator=''):
    filename = Path(filename)
    mkdir(filename, level=2, create_self=True)
    filename = str(filename / time.strftime('%Y-%m-%d__%H-%M-%S.txt'))

    logging.StreamHandler.terminator = terminator

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # logFormat = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    logFormat = logging.Formatter('%(message)s')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormat)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(filename=filename, mode=file_mode)
    fileHandler.setFormatter(logFormat)
    logger.addHandler(fileHandler)
    return logger


def get_img_target(img, target):
    if isinstance(target, (list, tuple)):
        target = target[0]
    if len(img.size()) == 5:
        B, D, C, H, W = img.size()
        img = img.view(B * D, C, H, W)
        target = target.view(B * D, 1, H, W)
    return img, target


def all_reduce(tensor):
    if isinstance(tensor, (tuple, list)):
        for t in tensor:
            all_reduce(t)
    else:
        dist.all_reduce(tensor)


def all_gather(tensorlist, tensor):
    if isinstance(tensor, (tuple, list)):
        for l, t in zip(tensorlist, tensor):
            all_gather(l, t)
    else:
        dist.all_gather(tensorlist, tensor)