from src.torchvahadane.optimizers import ista, coord_descent
import torch
import numpy as np
from src.torchvahadane.dict_learning import dict_learning
from src.torchvahadane.utils import get_concentrations, convert_OD_to_RGB, percentile, _to_tensor
from src.torchvahadane.stain_extractor_cpu import StainExtractorCPU
from src.torchvahadane.stain_extractor_gpu import StainExtractorGPU
from src.torchvahadane.histogram_matching import _fit_reference, _match_histograms_torch


class TorchVahadaneNormalizer():
    """
    GPU accelerated Vahadane normalization

    Source code adapted from:
    https://github.com/Peter554/StainTools/blob/master/staintools/stain_normalizer.py
    Uses spams as dependency for staintools like stain matrix estimation.
    If direct usage on gpu, idea is possibility to use nvidia based loadings with gpu decompression of wsi
    TorchVahadaneNormalizer also additionally adds the option to match the histogram of the vahadane normalized image to the reference image,
    which can add improved contrast and saturation to the normalization results.
    """

    def __init__(self, device='cuda', staintools_estimate=True, correct_exposure=False):
        """
        Parameters
        ----------
        device : str
            torch device in string form
        staintools_estimate : bool
            Default True. Using more robust and faster legacy spams based stain matrix estimation.
        correct_exposure : bool
            Default False. Perform histogram matching to target object after normalization during transformation.
            Can add improved contrast and saturation to the normalization results.
        """
        super().__init__()
        self.stain_matrix_target = None
        self.maxC_target = None
        self.stain_m_fixed = None
        self.device = torch.device(device)
        self.staintools_estimate = staintools_estimate
        self.stain_extractor = StainExtractorCPU() if self.staintools_estimate else StainExtractorGPU()
        self.correct_exposure = correct_exposure
        self.template_hist_fit = None

    def fit(self, target, method='ista'):
        """Short summary.

        Parameters
        ----------
        target : numpy.ndarray
            reference image.
        method : str
            Default 'ista'. optimizer method used for l1 regression.
        """

        mask = self.stain_extractor.get_tissue_mask(target)
        if self.staintools_estimate:
            self.stain_matrix_target = self.stain_extractor.get_stain_matrix(target, mask=mask)
            self.stain_matrix_target = _to_tensor(self.stain_matrix_target, self.device)
            target = _to_tensor(target, self.device)
        else:
            target = _to_tensor(target, self.device)
            self.stain_matrix_target = self.stain_extractor.get_stain_matrix(target, mask=mask)
        concentration_target = get_concentrations(target, self.stain_matrix_target, method=method)
        self.maxC_target = percentile(concentration_target.T, 99, dim=0)
        if self.correct_exposure:
            self.template_hist_fit = _fit_reference(target, mask=mask)

    def set_stain_matrix(self, stain_matrix):
        """set fixed stain matrix for transformation.

        Parameters
        ----------
        stain_matrix : numpy.ndarray
            stain instensites
        """
        self.stain_m_fixed = _to_tensor(stain_matrix, self.device)
        # print(self.stain_m_fixed, 'set as transform matrix')

    def set_stain_matrix_from_img(self, I):
        """set fixed stain matrix for transformation based on image

        Parameters
        ----------
        I : numpy.ndarray
        """
        if self.staintools_estimate:
            self.stain_m_fixed = self.stain_extractor.get_stain_matrix(I, mask=mask)
            self.stain_m_fixed = _to_tensor(self.stain_m_fixed, self.device)
        else:
            I = _to_tensor(I, self.device)
            self.stain_m_fixed = self.stain_extractor.get_stain_matrix(I)
        # print(self.stain_m_fixed, 'set as transform matrix')

    def transform(self, I, method='ista', r=0.01, return_mask=False):
        """Transform input image to reference image pased in fit using vahadane normalization.

        Parameters
        ----------
        I : numpy.ndarray
        method : str
            select other implemented optimer method.
        r : float
            l1-regularizer
        return_mask : bool
            additionally return the estimated tile tissue mask.
        Returns
        -------
        torch.Tensor
            normalized image
        """
        mask = self.stain_extractor.get_tissue_mask(I) if return_mask or self.correct_exposure or (
                    self.stain_m_fixed is None) else None
        if self.stain_m_fixed is None:
            if self.staintools_estimate:
                stain_matrix = self.stain_extractor.get_stain_matrix(I, mask=mask)
                stain_matrix = _to_tensor(stain_matrix, self.device)
            else:
                I = _to_tensor(I, self.device)
                stain_matrix = self.stain_extractor.get_stain_matrix(I, mask=mask)
        else:
            stain_matrix = self.stain_m_fixed

        I = _to_tensor(I, self.device)
        concentrations = get_concentrations(I, stain_matrix, regularizer=r, method=method)
        maxC = percentile(concentrations.T, 99, dim=0)
        concentrations *= (self.maxC_target / maxC)[:, None]
        out = 255 * torch.exp(-1 * torch.matmul(concentrations.T, self.stain_matrix_target))
        out = out.reshape(I.shape).type(torch.uint8)

        if self.correct_exposure:
            out = _match_histograms_torch(out, self.template_hist_fit, channel_axis=2,
                                          mask=_to_tensor(mask, self.device))

        if return_mask:
            return out, mask

        # add
        out = out.detach().cpu().numpy()

        return out


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    target = cv2.imread('test_images/TCGA-33-4547-01Z-00-DX7.91be6f90-d9ab-4345-a3bd-91805d9761b9_8270_5932_0.png')
    target = cv2.cvtColor(target, 4)
    norm = cv2.imread('test_images/TCGA-95-8494-01Z-00-DX1.716299EF-71BB-4095-8F4D-F0C2252CE594_5932_5708_0.png')
    norm = cv2.cvtColor(norm, 4)
    plt.imshow(norm)
    norm[2500:5000, 2500:5000, :] = 255
    #
    # # test implementation
    n = TorchVahadaneNormalizer(correct_exposure=True)
    n.fit(target)
    # self = n;I=norm; method='ista'; r=0.01  # # DEBUG:
    # %time n.transform(norm)
    plt.imshow(normalized_img.cpu()[100:200, 100:200, :])
    plt.imshow(normalized_img.cpu())
    plt.imshow(norm)

    n.correct_exposure = False
    normalized_img_ = n.transform(norm)
    plt.imshow(normalized_img_.cpu())

    #
    # ## stain tools
    # import staintools
    # n = staintools.StainNormalizer('vahadane')
    # n.fit(target)
    # normalized_img = n.transform(norm)
    # plt.imshow(normalized_img)
    # plt.imshow(norm)
    #
    # # test implementation gpu only
    # norm = torch.from_numpy(norm)
    # target = torch.from_numpy(target)
    # n = TorchVahadaneNormalizer(staintools_estimate=False)
    # n.fit(target)
    # normalized_img = n.transform(norm)
    # plt.imshow(normalized_img.cpu())
