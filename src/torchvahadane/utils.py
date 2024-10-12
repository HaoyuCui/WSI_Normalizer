"""
Parts directly adapted from https://github.com/Peter554/StainTools and https://github.com/EIDOSLAB/torchstain
"""
import torch
import numpy as np
from src.torchvahadane.optimizers import ista, coord_descent
from typing import Union


def is_cuda_ready():
    return torch.cuda.is_available()


class TissueMaskException(Exception):
    """generic Exception raised when calculations fail because of not enough foreground pixels"""
    pass


def _to_tensor(m, device=None):
    """if numpy turn to tensor, otherwise no-op"""
    m = m if isinstance(m, torch.Tensor) else torch.from_numpy(m).to(device)
    return m.float() if torch.is_floating_point(m) else m # we dont want float64

def _to_array(m):
    """if numpy turn to tensor, otherwise no-op"""
    return m.cpu().numpy() if isinstance(m, torch.Tensor) else m


def percentile(t: torch.Tensor, q: float, dim: int) -> Union[int, float]:
    """
    Author: addapted from https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30

    Return the ``q``-th percentile of the flattenepip d input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.shape[dim] - 1))  # interpolation?
    return t.kthvalue(k, dim=0).values


def convert_RGB_to_OD_cpu(I):
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = (I == 0)
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255), 1e-6)


def convert_RGB_to_OD(I):
    """
    adapted from: https://github.com/EIDOSLAB/torchstain
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = (I == 0)
    I[mask] = 1
    # I = torch.clamp_min_(I, 1)
    return torch.maximum(-1 * torch.log(I / 255), torch.tensor(1e-6))


def convert_OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB.
    adapted from: https://github.com/EIDOSLAB/torchstain
    RGB = 255 * exp(-1*OD_RGB)
    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, "Negative optical density."
    OD = torch.maximum(OD, 1e-6)
    return (255 * torch.exp(-1 * OD)).astype(torch.uint8)


def get_concentrations(I, stain_matrix, regularizer=0.01, method='ista'):
    """
    Estimate concentration matrix given an image and stain matrix.

    :param I:
    :param method: str, select optimizer method.
    :param stain_matrix:
    :param regularizer:
    :return:
    """
    OD = convert_RGB_to_OD(I).reshape((-1, 3)).to(I.device)
    if method == 'ista':
        return ista(OD, 'ridge', stain_matrix.T, alpha=regularizer, positive=True).T
    else:
        print(method, ' is not a valid optimizer')
        raise NotImplementedError
