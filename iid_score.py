import torch
from torch.nn.functional import pdist
import numpy as np


def prepare_1d_samples(activations_x1, activations_x2, activations_y1, activations_y2, device='cpu'):
    # Compute h(X1,X2), h(X1,Y1), h(Y1,Y2) samples (h - euclidean metric)
    activations_x1 = activations_x1.to(device, dtype=torch.float32)
    activations_y1 = activations_y1.to(device, dtype=torch.float32)
    activations_x2 = activations_x2.to(device, dtype=torch.float32)
    activations_y2 = activations_y2.to(device, dtype=torch.float32)
    
    data_xy = torch.flatten(torch.cdist(activations_x1, activations_y1)).to(device, dtype=torch.float16)
    data_xx = torch.flatten(torch.cdist(activations_x1, activations_x2)).to(device, dtype=torch.float16)
    data_yy = torch.flatten(torch.cdist(activations_y1, activations_y2)).to(device, dtype=torch.float16)
                
    return data_xx, data_yy, data_xy


def cramer_iid_score(p, data_xx, data_yy, data_xy, include_third_term=False, device='cpu'):
    """
    Calculates the IID for 1-dimensional statistical distance d being p-Cramer distance. 
    
    Params:
    -- p      : Order of the Cramer distance
    -- data_xx: Sample of the random variable h(X1,X2)
    -- data_yy: Sample of the random variable h(Y1,Y2)
    -- data_xy: Sample of the random variable h(X1,Y1)
    -- include_third_term: True is the term d(h(Y1,Y2), h(X1,Y1)) should be included,
                           False otherwise
    
    Returns:
    --   : The IID for 1-dimensional statistical distance d being p-Cramer distance.
    """
    dist1 = _cdf_distance(p, data_xx, data_yy, device)
    dist2 = _cdf_distance(p, data_xx, data_xy, device)
    
    if include_third_term:
        dist3 = _cdf_distance(p, data_xy, data_yy, device)
    else:
        dist3 = 0
    
    return dist1 + dist2 + dist3


def _cdf_distance(p, u_values, v_values, device='cpu'):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    u_values = u_values.to(device, dtype=torch.float16)
    v_values = v_values.to(device, dtype=torch.float16)
    
    u_sorter = torch.argsort(u_values).to(device)
    v_sorter = torch.argsort(v_values).to(device)

    all_values = torch.concatenate((u_values, v_values)).to(device, dtype=torch.float16)
    all_values = all_values.sort().values.to(device, dtype=torch.float16)

    # Compute the differences between pairs of successive values of u and v.
    deltas = torch.diff(all_values).to(device)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], side='right').to(device)
    v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], side='right').to(device)

    # Calculate the CDFs of u and v 
    u_cdf = u_cdf_indices / len(u_values)
    v_cdf = v_cdf_indices / len(v_values)

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return torch.sum(torch.multiply(torch.abs(u_cdf - v_cdf), deltas)).item()
    if p == 2:
        return torch.sum(torch.multiply(torch.square(u_cdf - v_cdf), deltas)).item() # no-root version
    return torch.sum(torch.multiply(torch.power(torch.abs(u_cdf - v_cdf), p), deltas)).item() # version without torch.power(..., 1/p)