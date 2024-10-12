""" Code directly adapted from https://github.com/Peter554/StainTools"""

from src.torchvahadane.utils import convert_RGB_to_OD, convert_RGB_to_OD_cpu, TissueMaskException
import cv2
import numpy as np
import spams


class StainExtractorCPU():

    def __init__(self):
        pass

    def get_tissue_mask(self, I, luminosity_threshold=0.8, custom_tissue_mask=None):
        """
        Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        Typically we use to identify tissue in the image and exclude the bright white background.

        :param I: RGB uint 8 image.
        :param luminosity_threshold: Luminosity threshold.
        :return: Binary mask.
        """
        I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
        mask = (L < luminosity_threshold) & (L > 0)  # fix a bug where black background in wsi are not ignored
        # mask = mask & custom_tissue_mask
        # Check it's not empty
        if mask.sum() == 0:
            raise TissueMaskException("Empty tissue mask computed")
        return mask

    def normalize_matrix_rows(self, A):
        """
        Normalize the rows of an array.

        :param A: An array.
        :return: Array with rows normalized.
        """
        return A / np.linalg.norm(A, axis=1)[:, None]

    def get_stain_matrix(self, I, luminosity_threshold=0.8, regularizer=0.1, mask=None):
        """
        Stain matrix estimation via method of:
        A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'

        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param regularizer:
        :return:
        """
        assert I.dtype == np.uint8, "Image should be RGB uint8."
        if mask is not None:
            assert mask.ndim == 2, 'Mask should not have channels'
        # convert to OD and ignore background
        tissue_mask = self.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,)) if mask is None else mask.reshape((-1,))
        OD = convert_RGB_to_OD_cpu(I).reshape((-1, 3))
        OD = OD[tissue_mask]
        # do the dictionary learning
        dictionary = spams.trainDL(X=OD.T, K=2, lambda1=regularizer, mode=2,
                                   modeD=0, posAlpha=True, posD=True, verbose=False).T
        # order H and E.
        # H on first row.
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]

        return self.normalize_matrix_rows(dictionary)
