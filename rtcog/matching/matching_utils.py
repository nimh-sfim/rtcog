import numpy as np


def pearson_correlations(data, template_centered, template_norms):
    """
    Compute spatial Pearson correlations between one vector and many templates.

    Template centering and norms are accepted as precomputed inputs so callers
    can reuse them across real-time volumes.

    Parameters
    ----------
    data : array_like
        Spatial data vector with shape ``(n_voxels,)``.
    template_centered : array_like
        Mean-centered templates with shape ``(n_templates, n_voxels)``.
    template_norms : array_like
        Euclidean norm of each centered template with shape ``(n_templates,)``.

    Returns
    -------
    np.ndarray
        Pearson correlation for each template. Constant or non-finite inputs
        receive a correlation of zero.
    """
    data = np.asarray(data, dtype=np.float32).ravel()
    template_centered = np.asarray(template_centered, dtype=np.float32)
    template_norms = np.asarray(template_norms, dtype=np.float32).ravel()

    data_centered = data - data.mean()
    data_norm = np.linalg.norm(data_centered)
    if data_norm == 0 or not np.isfinite(data_norm):
        return np.zeros(template_norms.size, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = (
            (template_centered @ data_centered) / (template_norms * data_norm)
        ).astype(np.float32)

    correlations[~np.isfinite(correlations)] = 0
    return np.clip(correlations, -1, 1)


def nmi_n_bins(n_voxels):
    """
    Return the NMI bin count for a spatial vector.

    Parameters
    ----------
    n_voxels : int
        Number of voxels in the vector being binned.

    Returns
    -------
    int
        The ceiling of the cube root of ``n_voxels``.
    """
    return int(np.ceil(n_voxels ** (1 / 3)))


def nmi_bin_data(data, n_bins=None):
    """
    Bin a spatial vector for normalized mutual information.

    Port of ``NMI_binData.m`` from gRAICAR.

    Parameters
    ----------
    data : array_like
        Spatial template or volume vector.
    n_bins : int, optional
        Number of bins to use. When omitted, ``ceil(n_voxels ** (1 / 3))`` is
        used.

    Returns
    -------
    np.ndarray
        One-based integer bin labels with shape ``(n_voxels,)``.

    Raises
    ------
    ValueError
        If ``data`` is empty.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n_voxels = data.size
    if n_voxels == 0:
        raise ValueError("Cannot bin an empty vector")

    if n_bins is None:
        n_bins = nmi_n_bins(n_voxels)

    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        # Constant vector: single bin
        return np.ones(n_voxels, dtype=np.int16)

    delta = (data_max - data_min) / n_voxels
    lower = data_min - delta / 2
    upper = data_max + delta / 2
    scaled = (data - lower) / (upper - lower) * n_bins + 0.5

    # MATLAB-style rounded bin coordinates
    bins = np.floor(scaled + 0.5).astype(np.int16)

    # Restrict to MATLAB-style 1-based bin labels
    return np.clip(bins, 1, n_bins)


def nmi_from_bins(x_bins, y_bins, n_bins=None):
    """
    Compute normalized mutual information for binned vectors.
    
    Port of ``calcNMI.m`` from gRAICAR.

    The estimate is ``(Hx + Hy) / Hxy``. Callers that need the MICM-style score
    used by the online NMI matcher should subtract one from this value.

    Parameters
    ----------
    x_bins, y_bins : array_like
        One-based integer bin labels for the two vectors being compared.
    n_bins : int, optional
        Number of bins in each vector. When omitted, the maximum bin label in
        either input is used.

    Returns
    -------
    float
        Normalized mutual information estimate.

    Raises
    ------
    ValueError
        If either input is empty after flattening.
    """
    x_bins = np.asarray(x_bins).ravel()
    y_bins = np.asarray(y_bins).ravel()
    total = min(x_bins.size, y_bins.size)
    if total == 0:
        raise ValueError("Cannot compute NMI for empty vectors")

    x_bins = x_bins[:total].astype(np.int64)
    y_bins = y_bins[:total].astype(np.int64)

    if n_bins is None:
        n_bins = int(max(np.max(x_bins), np.max(y_bins)))

    # Joint-histogram indices
    x_idx = np.clip(x_bins, 1, n_bins) - 1
    y_idx = np.clip(y_bins, 1, n_bins) - 1
    flat_idx = x_idx * n_bins + y_idx

    hist = np.bincount(flat_idx, minlength=n_bins * n_bins).reshape(n_bins, n_bins)
    hist = hist.astype(np.float64) / total

    eps = np.finfo(np.float64).eps
    hx = np.sum(hist, axis=1)
    hy = np.sum(hist, axis=0)

    # Marginal and joint entropies from calcNMI.m
    hx = -np.sum(hx * np.log2(hx + eps))
    hy = -np.sum(hy * np.log2(hy + eps))
    hxy = -np.sum(hist * np.log2(hist + eps)) + eps

    return (hx + hy) / hxy

def rt_svrscore_vol(data, SVRs, caps_labels):
    """
    Compute SVR scores using pretrained models.

    Parameters
    ----------
    data : np.ndarray
        The input data to be used for making predictions.
    SVRs : dict
        A dictionary of trained Support Vector Regressor (SVR) models, where the keys are 
        label names and the values are the corresponding SVR models.
    caps_labels : list of str
        A list of labels corresponding to the SVRs in `SVRs`. The function will use these 
        labels to predict the values from the respective SVRs.

    Returns
    -------

    np.ndarray
        The predicted values from each SVR for each label.
    """
    out = []

    for cap_lab in caps_labels:
        out.append(SVRs[cap_lab].predict(data[:,np.newaxis].T)[0])

    return np.array(out)[:,np.newaxis]

def rt_maskscore_vol(data, inputs, labels):
    out = []
    masked_templates = inputs["masked_templates"].item()
    masks = inputs["masks"].item()
    voxel_counts = inputs["voxel_counts"].item()

    for name in labels:
        mask = masks[name]
        template = masked_templates[name]
        masked_data = data[mask]        
        out.append(np.dot(template, masked_data) / voxel_counts[name])
    return np.array(out)[:, np.newaxis]
