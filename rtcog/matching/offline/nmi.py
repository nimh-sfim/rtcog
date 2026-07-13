import argparse
import logging
import os.path as osp

import numpy as np

from rtcog.matching.matching_utils import nmi_bin_data, nmi_n_bins
from rtcog.utils.core import file_exists
from rtcog.utils.fMRI import load_fMRI_file, mask_fMRI_img


log = logging.getLogger("offline_nmi")
log_fmt = logging.Formatter('[%(levelname)s - offline_nmi]: %(message)s')
log_ch = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)

#TODO: add offline NMI to help user establish threshold

def load_template_labels(labels_path, n_templates):
    # TODO: reuse this fn for other methods
    """
    Load template labels for an NMI template file.

    Parameters
    ----------
    labels_path : str or None
        Path to a comma-separated label file. When ``None``, labels default to
        ``T01``, ``T02``, and so on.
    n_templates : int
        Expected number of labels.

    Returns
    -------
    list of str
        Template labels in file order.

    Raises
    ------
    RuntimeError
        If ``labels_path`` cannot be read.
    ValueError
        If the number of labels does not match ``n_templates``.
    """
    if labels_path is None:
        return [f"T{i + 1:02d}" for i in range(n_templates)]

    try:
        with open(labels_path, "r") as f:
            labels = [label.strip() for label in f.read().strip().split(",") if label.strip()]
    except Exception as e:
        raise RuntimeError(f"Error loading template labels from {labels_path}: {e}")

    if len(labels) != n_templates:
        raise ValueError(f"Found {len(labels)} labels for {n_templates} templates")
    return labels


def build_nmi_template_data(templates_path, mask_path, template_labels_path=None):
    """
    Build an NMI matcher template dictionary from template maps.

    Parameters
    ----------
    templates_path : str
        Path to a 3D or 4D NIfTI image containing one or more template maps.
    mask_path : str
        Path to the mask image used to vectorize templates.
    template_labels_path : str, optional
        Path to comma-separated labels. When omitted, labels default to
        ``T01``, ``T02``, and so on.

    Returns
    -------
    dict
        Dictionary with ``labels``, raw ``templates``, precomputed
        ``template_bins``, and ``n_bins``. Raw templates are kept because the
        online matcher uses them for signed-correlation scoring.
    """
    templates_img = load_fMRI_file(templates_path)
    mask_img = load_fMRI_file(mask_path)

    masked = mask_fMRI_img(templates_img, mask_img)
    if masked.ndim == 1:
        templates = masked[np.newaxis, :]
    else:
        templates = masked.T

    templates = templates.astype(np.float32)
    n_templates, n_voxels = templates.shape
    labels = load_template_labels(template_labels_path, n_templates)
    n_bins = nmi_n_bins(n_voxels)
    template_bins = np.vstack([
        nmi_bin_data(template, n_bins) for template in templates
    ])

    return {
        "labels": np.array(labels),
        "templates": templates,
        "template_bins": template_bins,
        "n_bins": np.array(n_bins),
    }


def save_nmi_template_data(template_data, out_dir, prefix):
    """
    Save an NMI matcher template dictionary.

    Parameters
    ----------
    template_data : dict
        Template dictionary returned by :func:`build_nmi_template_data`.
    out_dir : str
        Existing output directory.
    prefix : str
        Filename prefix for the saved ``.nmi_templates.npz`` file.

    Returns
    -------
    str
        Path to the saved template file.

    Raises
    ------
    FileNotFoundError
        If ``out_dir`` does not exist.
    """
    if not osp.isdir(out_dir):
        raise FileNotFoundError(f"Out directory does not exist: {out_dir}")

    out_path = osp.join(out_dir, f"{prefix}.nmi_templates.npz")
    np.savez(out_path, **template_data)
    log.info(f"Saved NMI template data to: {out_path}")
    return out_path


def process_options():
    parser = argparse.ArgumentParser(description="Prepare template maps for real-time NMI matching")
    parser_inopts = parser.add_argument_group("Input Options", "Inputs to this program")
    parser_inopts.add_argument(
        "-t", "--templates_path",
        help="Path to template maps",
        dest="templates_path",
        action="store",
        type=file_exists,
        required=True,
    )
    parser_inopts.add_argument(
        "-m", "--mask",
        action="store",
        type=file_exists,
        dest="mask_path",
        help="Path to mask",
        required=True,
    )
    parser_inopts.add_argument(
        "-l", "--template_labels_path",
        help="Path to comma-separated template labels; defaults to T01, T02, ...",
        dest="template_labels_path",
        action="store",
        type=file_exists,
        default=None,
    )
    parser_outopts = parser.add_argument_group("Output Options", "Where to save results")
    parser_outopts.add_argument(
        "-o", "--out_dir",
        action="store",
        type=str,
        dest="out_dir",
        default="./",
        help="Output directory [Default: %(default)s]",
    )
    parser_outopts.add_argument(
        "-p", "--prefix",
        action="store",
        type=str,
        dest="prefix",
        default="nmi",
        help="Output prefix [Default: %(default)s]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = process_options()
    template_data = build_nmi_template_data(
        opts.templates_path,
        opts.mask_path,
        opts.template_labels_path,
    )
    save_nmi_template_data(template_data, opts.out_dir, opts.prefix)
