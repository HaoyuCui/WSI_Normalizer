"""
@author Cédric Walker
Naive grid search on wsi tiles to extract the median stain intensities.
TorchVahadaneNormalizer accepts a fixed stain matrix with normalizer.set_stain_matrix()
This also speeds up subsequent transform() calls, as the stain matrix is not reestimated from image.
"""
import cv2
import torch
import openslide
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from src.torchvahadane.utils import TissueMaskException

def _run(coord, osh_level, tile_size):
    """internal helper function"""
    return  np.ascontiguousarray(osh_.read_region(coord, osh_level, (tile_size, tile_size)))[:,:,0:3].copy()

def _check_mask_bounds(x,y, mask, tile_size, ds_m, thresh=0.5):
    """internal helper function to check wether tile coords contain tissue"""
    t_s_mask = round(tile_size/ds_m)
    return ((x >= mask.shape[1] or y >= mask.shape[0]) or mask[y:y+t_s_mask,x:x+t_s_mask].mean() < thresh)

def estimate_median_matrix(osh, normalizer, osh_level, mask=None, tile_size=4096, stride=None, mask_ds=16, num_workers=4):
    """Estimating median stain intensity matrix based on a grid search as proposed in Vahadane et al.
    TorchVahadaneNormalizer accepts a fixed stain matrix with normalizer.set_stain_matrix(), where the stain matrix does not get
    reestimated for every source image. This also speeds up subsequent normalizer.transform() calls
    Parameters
    ----------
    osh : openslide.OpenSlide
    normalizer : torchvahadane.TorchVahadaneNormalizer
    osh_level : int
        openslide level to extract tiles for stain matrix estimation on.
        Corresponds to tissue resolution/magnification.
    tile_size : int
        tile_size of image tiles extracted for stain matrix estimation on.
    mask : numpy.ndarray
        optional tissue mask for wsi. Regions not within tissue mask are excluded from stain estimation.
    stride : int
        optional stride parameter for grid search.
    mask_ds : int
        optional parameter to resize the mask parameter to a openslide level, default 16.
    num_workers : int
        mumber of processed and threads used to accelerate grid search.

    Returns
    -------
    numpy.ndarray
        stain matrix estimated from WSI using median intensities over tiles.
    """
    global osh_
    osh_ = osh

    stride = tile_size if stride is None else stride
    m_level = osh.get_best_level_for_downsample(mask_ds + .5)

    if mask is None:
        img = osh.read_region((0, 0), m_level, osh.level_dimensions[m_level])
        mask = normalizer.stain_extractor.get_tissue_mask(np.asarray(img)[:,:,:3])

    m_size=osh.level_dimensions[m_level]
    ds_m = round(osh.level_downsamples[m_level], 1)
    mask = cv2.resize(mask.astype(np.uint8), m_size, cv2.INTER_NEAREST).astype(bool)
    for thresh in [0.5, 0.25, 0.125]:  # for small wsi, automatically lower the threshold if no tiles where found.
        coords = [(round(x*ds_m),round(y*ds_m))
                    for y in range(0,osh.level_dimensions[m_level][1], round(stride/ds_m))
                        for x in range(0,osh.level_dimensions[m_level][0], round(stride/ds_m))
                            if not _check_mask_bounds(x,y, mask, tile_size, ds_m, thresh)]
        if coords:
            break
    else:
        raise TissueMaskException('Not enough tissue area found for calculating stains. Try lowering the tile_size')

    with Pool(num_workers) as pool:
        matrix = list(pool.imap(partial(_run, osh_level=osh_level, tile_size=tile_size), coords, chunksize=1))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        matrix = list(executor.map(normalizer.stain_extractor.get_stain_matrix, matrix, chunksize=1))
    return np.median(np.array(matrix), axis=(0))/np.linalg.norm(np.median(np.array(matrix), axis=(0)), axis=1)[:,None]



if __name__ == '__main__':
    osh = openslide.open_slide('/data/kettering_ov/imgs/4172691.svs')

    lvl_mask=2
    ds_mask=16
    img = osh.read_region((0, 0), lvl_mask, osh.level_dimensions[lvl_mask])
    import matplotlib.pyplot as plt
    from torchvahadane import TorchVahadaneNormalizer
    normalizer = TorchVahadaneNormalizer()
    estimate_median_matrix(osh, normalizer,0, None)

    img = cv2.resize(np.asarray(img)[:, :, 0:3], dsize=(0, 0), fx=ds_mask/8, fy=ds_mask/8)
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th, _ = cv2.threshold(imgg, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
    th = max(220, th)
    mask = np.bitwise_and(imgg > 0, imgg < th)

    estimate_median_matrix(osh, normalizer,0, mask)
    plt.imshow(mask)
