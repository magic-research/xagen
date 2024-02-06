# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    stats_real, stats_real_f, stats_real_rh, stats_real_lh = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real)
    mu_real, sigma_real = stats_real.get_mean_cov()
    mu_real_f, sigma_real_f = stats_real_f.get_mean_cov()
    mu_real_rh, sigma_real_rh = stats_real_rh.get_mean_cov()
    mu_real_lh, sigma_real_lh = stats_real_lh.get_mean_cov()

    stats_gen, stats_gen_f, stats_gen_rh, stats_gen_lh = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen)
    mu_gen, sigma_gen = stats_gen.get_mean_cov()
    mu_gen_f, sigma_gen_f = stats_gen_f.get_mean_cov()
    mu_gen_rh, sigma_gen_rh = stats_gen_rh.get_mean_cov()
    mu_gen_lh, sigma_gen_lh = stats_gen_lh.get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    m = np.square(mu_gen_f - mu_real_f).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen_f, sigma_real_f), disp=False) # pylint: disable=no-member
    fid_f = np.real(m + np.trace(sigma_gen_f + sigma_real_f - s * 2))    

    m = np.square(mu_gen_rh - mu_real_rh).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen_rh, sigma_real_rh), disp=False) # pylint: disable=no-member
    fid_rh = np.real(m + np.trace(sigma_gen_rh + sigma_real_rh - s * 2))    

    m = np.square(mu_gen_lh - mu_real_lh).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen_lh, sigma_real_lh), disp=False) # pylint: disable=no-member
    fid_lh = np.real(m + np.trace(sigma_gen_lh + sigma_real_lh - s * 2))    
    print(fid, fid_f, fid_rh, fid_lh)
    return (fid, fid_f, fid_rh, fid_lh)

#----------------------------------------------------------------------------
