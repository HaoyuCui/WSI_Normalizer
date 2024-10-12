import torch
from src.torchvahadane.utils import convert_RGB_to_OD, convert_RGB_to_OD_cpu, TissueMaskException
import cv2
import numpy as np
from src.torchvahadane.dict_learning import dict_learning
use_kornia=True
try:
    from kornia.color import rgb_to_lab
except ImportError:
    use_kornia = False

class StainExtractorGPU():

    def __init__(self):
        if not use_kornia:
            print('Kornia not installed. Using cv2 fallback')


    def get_tissue_mask(self, I, luminosity_threshold=0.8):
        """
        Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        Typically we use to identify tissue in the image and exclude the bright white background.

        uses kornia as optional dependency
        :param I: RGB uint 8 image.
        :param luminosity_threshold: Luminosity threshold.
        :return: Binary mask.
        """
        if use_kornia:
            I_LAB = rgb_to_lab(I[None,:,:,:].transpose(1,3)/255)
            L = (I_LAB[:, 0,:,:] /100 ).squeeze() # Convert to range [0,1].
        else:
            I_LAB = torch.from_numpy(cv2.cvtColor(I.cpu().numpy(), cv2.COLOR_RGB2LAB))
            L = (I_LAB[:,:,0]/255).squeeze()
        # also check for rgb == 255!
        mask = (L < luminosity_threshold ) & (L > 0)  # fix bug in original stain tools code where black background is not ignored.
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
        return A / torch.linalg.norm(A, dim=1)[:, None]


    def get_stain_matrix(self, I, luminosity_threshold=0.8, regularizer=0.1, mask=None):
        """
        Stain matrix estimation via method of:
        A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'

        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param regularizer:
        :return:
        """
        # convert to OD and ignore background
        mask = self.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,)) if mask is None else  mask.reshape((-1,))
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[mask]
        # Change to pylasso dictionary training.
        dictionary, losses = dict_learning(OD, n_components=2, alpha=regularizer, lambd=0.1,
             algorithm='ista', device='cuda', steps=70, constrained=True, progbar=False, persist=True, init='ridge', cut=30, maxiter=50, positive=True)
        # H on first row.
        dictionary = dictionary.T
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        return self.normalize_matrix_rows(dictionary)
