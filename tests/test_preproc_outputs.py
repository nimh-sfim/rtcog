"""
This is for testing the final outputs of rtcaps_matcher.py given different parameters (preproc mode) against files created offline.
The process of generating the files is manual right now, but should be automated later.
"""
# -------------------------------
import sys
import os.path as osp
import pytest
import numpy as np
import nibabel as nib
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


from rtcaps.rtcap_lib.fMRI import mask_fMRI_img, unmask_fMRI_img
from rtcaps.rtcap_lib.rt_functions import init_EMA, rt_EMA_vol
from rtcaps.config import OUTPUT_DIR


def load_nii(fname):
    return nib.load(osp.join(OUTPUT_DIR, fname))

def load_arr(fname):
    return np.load(osp.join(OUTPUT_DIR, fname))

def get_metrics(arr1, arr2):
    arr1, arr2 = arr1.flatten(), arr2.flatten()
    rmse = mean_squared_error(arr1, arr2)
    corr, _ = pearsonr(arr1, arr2)

    return rmse, corr


def test_all_off(sample_data):
    pass


def test_snorm_only(sample_data):
    snorm_offline = load_nii('offline_preproc.snorm.nii')
    snorm_online = load_nii("online_preproc.pp_Zscore.snorm-on.nii")

    snorm_off_masked = mask_fMRI_img(snorm_offline, sample_data.mask_img)
    snorm_on_masked = mask_fMRI_img(snorm_online, sample_data.mask_img)

    rmse, corr = get_metrics(snorm_off_masked, snorm_on_masked)

    assert snorm_online.shape == snorm_offline.shape
    
    assert rmse < 0.1
    assert corr >= 0.93


def test_ema_only(sample_data):
    # Not the best test because I’m just using the rt functions instead of something else
    orig_data = load_arr("DataFromAFNI.end.npy")
    n = 0

    EMA_th, EMA_filt = init_EMA()

    offline_ema = []

    for t in range(10, orig_data.shape[1]):
        n += 1
        data_out, EMA_filt = rt_EMA_vol(n, EMA_th, orig_data[:, :t+1], EMA_filt, do_operation=True)
            
        offline_ema.append(data_out)

    offline_ema = np.concatenate(offline_ema, axis=-1)
    zeros = np.zeros((offline_ema.shape[0], 10))

    offline_ema = np.hstack([zeros, offline_ema])

    online_ema = mask_fMRI_img(load_nii('online_preproc.pp_EMA.nii').get_fdata(), sample_data.mask_img)

    rmse, corr = get_metrics(offline_ema, online_ema)

    assert offline_ema.shape == online_ema.shape
    # Need to come up with thresholds for this, maybe implement warnings instead of errors too?
    assert rmse < 0.1
    assert corr >= 0.93


def test_smooth_only(sample_data):
    pass


def test_iglm_only(sample_data):
    pass


def test_kalman_only(sample_data):
    pass


def test_all_on(sample_data):
    pass


def test_1d_motion(sample_data):
    pass


def test_blur(sample_data):
    pass

if __name__ == "__main__":
    pytest.main()