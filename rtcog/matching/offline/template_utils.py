"""Shared preparation helpers for spatial template matching methods."""

import argparse

import numpy as np

from rtcog.matching.matching_utils import pearson_correlations
from rtcog.utils.core import file_exists
from rtcog.utils.fMRI import load_fMRI_file, mask_fMRI_img


def load_template_labels(labels_path, n_templates):
    """Load comma-separated labels or create ``T01``, ``T02``, ... defaults."""
    if labels_path is None:
        return [f"T{i + 1:02d}" for i in range(n_templates)]

    try:
        with open(labels_path, "r", encoding="utf-8") as labels_file:
            labels = [
                label.strip()
                for label in labels_file.read().strip().split(",")
                if label.strip()
            ]
    except Exception as exc:
        raise RuntimeError(
            f"Error loading template labels from {labels_path}: {exc}"
        ) from exc

    if len(labels) != n_templates:
        raise ValueError(f"Found {len(labels)} labels for {n_templates} templates")
    return labels


def load_and_mask_image(image_path, mask_path=None, *, mask_img=None, verbose=False):
    """Load an image and return it, its mask image, and its masked data."""
    if mask_img is None:
        if mask_path is None:
            raise ValueError("Either mask_path or mask_img must be provided")
        mask_img = load_fMRI_file(mask_path, verbose=verbose)

    image_img = load_fMRI_file(image_path, verbose=verbose)
    masked = mask_fMRI_img(image_img, mask_img)
    return image_img, mask_img, masked


def load_template_maps(templates_path, mask_path, labels_path=None, *, verbose=False):
    """Load template maps, mask them, and validate their labels."""
    templates_img, mask_img, templates = load_and_mask_image(
        templates_path,
        mask_path,
        verbose=verbose,
    )
    templates = np.asarray(templates)
    if templates.ndim == 1:
        templates = templates[:, np.newaxis]
    if templates.ndim != 2:
        raise ValueError(
            "Masked templates must produce a 2D (n_voxels, n_templates) array"
        )

    labels = load_template_labels(labels_path, templates.shape[1])
    return templates_img, mask_img, templates, labels


def prepare_template_data(templates_path, mask_path, labels_path=None):
    """Load and mask template maps in template-by-voxel shape."""
    _, _, masked, labels = load_template_maps(
        templates_path,
        mask_path,
        labels_path,
    )
    templates = np.asarray(masked.T, dtype=np.float32)
    return {
        "labels": np.asarray(labels),
        "templates": templates,
    }


def load_masked_timeseries(data_path, mask_path):
    """Load processed data in voxel-by-time shape."""
    _, _, masked = load_and_mask_image(data_path, mask_path)
    data = np.asarray(masked, dtype=np.float32)
    return data[:, np.newaxis] if data.ndim == 1 else data


def score_pearson_timeseries(template_data, data, discard=0):
    """Score each time point against prepared templates with spatial Pearson r."""
    templates = np.asarray(template_data["templates"], dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if templates.ndim != 2 or data.ndim != 2:
        raise ValueError("Templates and data must both be 2D arrays")
    if templates.shape[1] != data.shape[0]:
        raise ValueError(
            f"Templates have {templates.shape[1]} voxels, but data has "
            f"{data.shape[0]}"
        )
    if discard < 0:
        raise ValueError("discard must be non-negative")

    centered = templates - templates.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    scores = np.zeros((templates.shape[0], data.shape[1]), dtype=np.float32)
    for timepoint in range(discard, data.shape[1]):
        scores[:, timepoint] = pearson_correlations(
            data[:, timepoint], centered, norms
        )
    return scores


def spatial_template_parser(
    method_name,
    default_prefix,
    *,
    description=None,
    data_required=False,
    labels_required=False,
    data_type=file_exists,
    mask_type=file_exists,
    out_dir_dest="out_dir",
):
    """Create a parser with the arguments shared by offline matchers."""
    if description is None:
        description = (
            f"Prepare template maps and optionally score processed data for "
            f"{method_name} matching"
        )
    labels_help = "Path to comma-separated template labels"
    if not labels_required:
        labels_help += "; defaults to T01, T02, ..."

    parser = argparse.ArgumentParser(description=description)
    inputs = parser.add_argument_group("Input Options", "Inputs to this program")
    inputs.add_argument(
        "-d",
        "--data",
        type=data_type,
        dest="data_path",
        default=None,
        required=data_required,
        help="Path to processed 4D data",
    )
    inputs.add_argument(
        "-t",
        "--templates_path",
        type=file_exists,
        required=True,
        help="Path to template maps",
    )
    inputs.add_argument(
        "-m",
        "--mask",
        type=mask_type,
        dest="mask_path",
        required=True,
        help="Path to the analysis mask",
    )
    inputs.add_argument(
        "-l",
        "--template_labels_path",
        type=file_exists,
        default=None,
        required=labels_required,
        help=labels_help,
    )
    inputs.add_argument(
        "--discard",
        type=int,
        dest="nvols_discard",
        default=100,
        help=(
            "Number of initial offline volumes to discard or leave unscored "
            "[Default: %(default)s]"
        ),
    )

    outputs = parser.add_argument_group("Output Options", "Where to save results")
    outputs.add_argument(
        "-o",
        "--out_dir",
        dest=out_dir_dest,
        default="./",
        help="Existing output directory [Default: %(default)s]",
    )
    outputs.add_argument(
        "-p",
        "--prefix",
        default=default_prefix,
        help="Output prefix [Default: %(default)s]",
    )
    return parser, inputs, outputs
