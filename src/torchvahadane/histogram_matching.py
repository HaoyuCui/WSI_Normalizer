import torch
from src.torchvahadane import TissueMaskException
from src.torchvahadane.utils import _to_tensor

def interp(x, xp, fp):
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
        from https://github.com/pytorch/pytorch/issues/50334
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)
    return m[indicies] * x + b[indicies]


def _match_cumulative_cdf(source, template, mask_s=None, mask_t=None):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    adapted from skimage.exposure
    """
    mask_s = _to_tensor(mask_s).reshape((-1)).type(torch.bool) if mask_s is not None else None
    mask_t = _to_tensor(mask_t).reshape((-1)).type(torch.bool) if mask_t is not None else None

    src_lookup = source.reshape(-1)
    r_tmp = src_lookup[mask_t].squeeze()  # if None introduces new dimension, so squeeze, otherwise select on mask index
    s_tmp = source.reshape(-1)[mask_s].squeeze()

    if source.dtype == torch.uint8:
        src_counts = torch.bincount(s_tmp, minlength=256)
        tmpl_counts = torch.bincount(r_tmp)
        # omit values where the count was 0
        tmpl_values = torch.nonzero(tmpl_counts).squeeze()
        tmpl_counts = tmpl_counts[tmpl_values]
        # src_values = torch.nonzero(src_counts).squeeze()
        # src_counts = src_counts[src_values]
    else:
        raise NotImplementedError('Only implemented for uint8')
    # calculate normalized quantiles for each array
    src_quantiles = torch.cumsum(src_counts,0) / s_tmp.numel()
    tmpl_quantiles = torch.cumsum(tmpl_counts,0) / r_tmp.numel()
    interp_a_values_ = torch.round(interp(src_quantiles, tmpl_quantiles.to(src_quantiles.device), tmpl_values.to(src_quantiles.device)))
    out = torch.round(torch.index_select(interp_a_values_, dim=0, index=src_lookup.long())).type(torch.uint8)
    out[~mask] = src_lookup[~mask]
    return out.reshape(source.shape)


def match_histograms_torch(image, reference, channel_axis=None, mask_s=None, mask_t=None):
    """ torchbased implementation of skimage.exposure.match_histograms which additionally
        accepts masks for input and reference image to match histogram only on foreground pixels

    Parameters
    ----------
    image : torch.Tensor
        Tensor in HXWXC or HXW format.
    reference : torch.Tensor
        Description of parameter `reference`.
    channel_axis : int
        channel axis for HXWXC image.
    mask_s : torch.Tensor or numpy.ndarray
        optional foreground mask for source image
    mask_t : torch.Tensor or numpy.ndarray
        optional foreground mask for reference image

    Returns
    -------
    type
        Description of returned object.

    """

    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number'
                     'of channels.')
    if channel_axis is not None:
        matched = torch.empty_like(image)
        for channel in range(image.shape[channel_axis]):
            matched_channel = _match_cumulative_cdf(image[..., channel],
                                                    reference[..., channel], mask_s=mask_s, mask_t=mask_t)
            matched[..., channel] = matched_channel
        return matched
    else:
        return _match_cumulative_cdf(image, reference).float()


def _fit_reference(reference, channel_axis=2, mask=None):
    """internal function to keep only histogram values in memory rather than whole reference image
        this is function is used within torchvahadane"""
    template_fit = []
    # mask = _to_tensor(mask).reshape((-1)).type(torch.bool) if mask is not None else None
    mask = mask.reshape((-1)).astype(bool) if mask is not None else None

    for channel in range(reference.shape[channel_axis]):
        r_tmp = reference[..., channel].reshape(-1)[mask].squeeze()
        tmpl_counts = torch.bincount(r_tmp)
        # omit values where the count was 0
        tmpl_values = torch.nonzero(tmpl_counts).squeeze()
        tmpl_counts = tmpl_counts[tmpl_values]
        tmpl_quantiles = torch.cumsum(tmpl_counts,0) / r_tmp.numel()
        template_fit.append((tmpl_quantiles, tmpl_values))
    return template_fit

def __match_cumulative_cdf(source, template_fit, mask=None):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    adapted from skimage
    """

    tmpl_quantiles, tmpl_values = template_fit
    mask = mask.reshape((-1)) if mask is not None else None
    if source.dtype == torch.uint8:
        src_lookup = source.reshape(-1)
        s_tmp = src_lookup[mask].squeeze()
        src_counts = torch.bincount(s_tmp, minlength=256)
        # src_values = torch.nonzero(src_counts).squeeze()
        # src_counts = src_counts[src_values]
    else:
        raise NotImplementedError('Only implemented for uint8')
    # calculate normalized quantiles for each array
    src_quantiles = torch.cumsum(src_counts,0) / s_tmp.numel()
    # interpolate only values in mask. values that are outside of mask, set to their own value
    interp_a_values_ = torch.round(interp(src_quantiles, tmpl_quantiles.to(src_quantiles.device), tmpl_values.to(src_quantiles.device)))
    out = torch.round(torch.index_select(interp_a_values_, dim=0, index=src_lookup.long())).type(torch.uint8)
    out[~mask] = src_lookup[~mask]
    return out.reshape(source.shape)

# def __match_cumulative_cdf(source, template_fit, mask=None): faster but has different artifacts.
#     tmpl_quantiles, tmpl_values = template_fit
#     mask = mask.reshape((-1)) if mask is not None else None
#     if source.dtype == torch.uint8:
#         s_tmp = source.reshape(-1)[mask].squeeze()
#         src_lookup = source.reshape(-1)
#         src_counts = torch.bincount(s_tmp, minlength=255)
#         src_values = torch.nonzero(src_counts).squeeze()
#         src_counts = src_counts[src_values]
#     else:
#         raise NotImplementedError('Only implemented for uint8')
#     src_quantiles = torch.cumsum(src_counts,0) / s_tmp.numel()
#     if src_quantiles.shape == torch.Size([]):
#         raise TissueMaskException('Not enough unique values in histogram calculation after applying tissue mask.') # or instead of interpolating just return value
#     interp_a_values_ = interp(src_quantiles, tmpl_quantiles.to(src_quantiles.device), tmpl_values.to(src_quantiles.device))
#     interp_a_values = torch.arange(0,256).float().to(src_quantiles.device)
#     interp_a_values[src_values] = interp_a_values_
#     return torch.index_select(interp_a_values, dim=0, index=src_lookup.long()).reshape(source.shape)


def _match_histograms_torch(image, reference_fit, channel_axis=2, mask=None):
    """internally used torchbased implementation of skimage.exposure.match_histograms
        accepts masks for input and reference to match histogram only on foreground pixels

        accepts fit reference rather than reference image
        """
    if channel_axis is not None:
        matched = torch.empty_like(image)
        for channel in range(image.shape[channel_axis]):
            matched_channel = __match_cumulative_cdf(image[..., channel],
                                                    reference_fit[channel], mask=mask)
            matched[..., channel] = matched_channel
        return matched
    else:
        return NotImplementedError('Only implemented for multichannel matching. Use match_histograms_torch instead')

if __name__ == '__main__':
    # torch.empty(io_.shape, dtype=io_.dtype, device=io_.device)

    ## quick test
    image = torch.randint(5,255,(5000,5000,3), dtype=torch.uint8).cuda()
    ref = torch.randint(4,255,(5000,5000,3), dtype=torch.uint8).cuda()
    mask_t = torch.all(ref<10, dim = 2) # torch.cat((torch.ones((50,20), dtype=bool), torch.zeros((50,30), dtype=bool)), dim=1)
    mask_s = torch.all(image<10, dim = 2) # torch.cat((torch.ones((50,20), dtype=bool), torch.zeros((50,30), dtype=bool)), dim=1)
    reference_fit = _fit_reference(ref)
    assert torch.all(_match_histograms_torch(image, reference_fit, channel_axis=2) == match_histograms_torch(image, ref, channel_axis=2)).item()

    reference_fit = _fit_reference(ref, mask=mask_t.cpu().numpy())
    assert torch.all(_match_histograms_torch(image, reference_fit, channel_axis=2, mask=mask_s) == match_histograms_torch(image, ref, channel_axis=2, mask_s = mask_s, mask_t=mask_t)).item()
