import argparse
import logging
import os.path as osp

import numpy as np
import pandas as pd
import hvplot.pandas
import panel as pn
import matplotlib.pyplot as plt

from rtcog.matching.matching_utils import nmi_bin_data, nmi_from_bins, nmi_n_bins
from rtcog.matching.offline.stats import (
    pairwise_template_stats,
    pairwise_trace_stats,
    pairwise_value_heatmap,
    save_stats_csvs,
)
from rtcog.utils.core import file_exists
from rtcog.utils.fMRI import load_fMRI_file, mask_fMRI_img


log = logging.getLogger("offline_nmi")
log_fmt = logging.Formatter('[%(levelname)s - offline_nmi]: %(message)s')
log_ch = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)


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


class OfflineNMI:
    def __init__(
        self,
        opts=None,
        *,
        templates_path=None,
        mask_path=None,
        template_labels_path=None,
        data_path=None,
        discard=100,
        out_dir="./",
        prefix="nmi",
    ):
        if opts is not None:
            templates_path = opts.templates_path
            mask_path = opts.mask_path
            template_labels_path = opts.template_labels_path
            data_path = opts.data_path
            discard = opts.nvols_discard
            out_dir = opts.out_dir
            prefix = opts.prefix

        self.templates_path = templates_path
        self.mask_path = mask_path
        self.template_labels_path = template_labels_path
        self.data_path = data_path
        self.discard = int(discard)
        self.out_dir = out_dir
        self.prefix = prefix
        self.out_path = osp.join(out_dir, prefix)

        self.template_data = None
        self.score_results = None

    @classmethod
    def from_template_data(cls, template_data, **kwargs):
        offline_nmi = cls(**kwargs)
        offline_nmi.template_data = template_data
        return offline_nmi

    @property
    def labels(self):
        return self.template_data["labels"].tolist()

    def build_template_data(self):
        templates_img = load_fMRI_file(self.templates_path)
        mask_img = load_fMRI_file(self.mask_path)

        masked = mask_fMRI_img(templates_img, mask_img)
        templates = masked[np.newaxis, :] if masked.ndim == 1 else masked.T
        templates = templates.astype(np.float32)

        n_templates, n_voxels = templates.shape
        labels = load_template_labels(self.template_labels_path, n_templates)
        n_bins = nmi_n_bins(n_voxels)

        self.template_data = {
            "labels": np.array(labels),
            "templates": templates,
            "template_bins": np.vstack([nmi_bin_data(template, n_bins) for template in templates]),
            "n_bins": np.array(n_bins),
        }
        return self.template_data

    def save_template_data(self):
        self._require_out_dir()
        self._require_template_data()

        out_path = f"{self.out_path}.nmi_templates.npz"
        np.savez(out_path, **self.template_data)
        log.info(f"Saved NMI template data to: {out_path}")
        return out_path

    def score_data(self):
        self._require_template_data()

        data_img = load_fMRI_file(self.data_path)
        mask_img = load_fMRI_file(self.mask_path)
        data = mask_fMRI_img(data_img, mask_img).astype(np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        scores, raw_scores, correlations = self._score_timepoints(data)
        self.score_results = {
            "labels": np.array(self.labels),
            "scores": scores,
            "raw_scores": raw_scores,
            "correlations": correlations,
            "traces": {label: scores[idx] for idx, label in enumerate(self.labels)},
            "raw_traces": {label: raw_scores[idx] for idx, label in enumerate(self.labels)},
            "correlation_traces": {label: correlations[idx] for idx, label in enumerate(self.labels)},
            "discard": self.discard,
        }
        return self.score_results

    def _score_timepoints(self, data):
        templates = np.asarray(self.template_data["templates"], dtype=np.float32)
        template_bins = np.asarray(self.template_data["template_bins"], dtype=np.int16)
        n_bins = int(np.asarray(self.template_data["n_bins"]).item())

        n_templates, n_voxels = templates.shape
        if data.shape[0] != n_voxels:
            raise ValueError(
                f"NMI templates have {n_voxels} voxels, but masked data has {data.shape[0]}"
            )

        n_timepoints = data.shape[1]
        scores = np.zeros((n_templates, n_timepoints), dtype=np.float32)
        raw_scores = np.zeros((n_templates, n_timepoints), dtype=np.float32)
        correlations = np.full((n_templates, n_timepoints), np.nan, dtype=np.float32)
        template_centered = templates - templates.mean(axis=1, keepdims=True)
        template_norms = np.linalg.norm(template_centered, axis=1)

        for tr in range(self.discard, n_timepoints):
            tr_data = data[:, tr].astype(np.float32).ravel()
            data_centered = tr_data - tr_data.mean()
            data_norm = np.linalg.norm(data_centered)
            if data_norm == 0:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                tr_correlations = (
                    (template_centered @ data_centered) / (template_norms * data_norm)
                ).astype(np.float32)
            correlations[:, tr] = tr_correlations

            valid = np.isfinite(tr_correlations) & (tr_correlations != 0)
            if not np.any(valid):
                continue

            data_bins = nmi_bin_data(tr_data, n_bins)
            for idx in np.flatnonzero(valid):
                raw_score = float(nmi_from_bins(data_bins, template_bins[idx], n_bins) - 1)
                raw_scores[idx, tr] = raw_score
                strength = max(raw_score, 0)
                scores[idx, tr] = strength if tr_correlations[idx] > 0 else -strength

        return scores, raw_scores, correlations

    def save_score_data(self):
        self._require_out_dir()
        if self.score_results is None:
            self.score_data()

        score_paths = {
            "scores": f"{self.out_path}.nmi_scores.npy",
            "raw_scores": f"{self.out_path}.nmi_raw_scores.npy",
            "correlations": f"{self.out_path}.nmi_correlations.npy",
            "traces": f"{self.out_path}.nmi_score_traces.npz",
        }

        np.save(score_paths["scores"], self.score_results["scores"])
        np.save(score_paths["raw_scores"], self.score_results["raw_scores"])
        np.save(score_paths["correlations"], self.score_results["correlations"])
        np.savez(score_paths["traces"], **self.score_results["traces"])

        log.info(f"Saved NMI scores to: {score_paths['scores']}")
        log.info(f"Saved raw NMI scores to: {score_paths['raw_scores']}")
        log.info(f"Saved NMI correlations to: {score_paths['correlations']}")
        log.info(f"Saved NMI score traces to: {score_paths['traces']}")
        return score_paths

    def build_template_stats_tables(self):
        self._require_template_data()
        return {
            "nmi_template_pairwise_stats": pairwise_template_stats(
                self.labels,
                self.template_data["templates"],
                self.template_data["templates"] != 0,
            ),
        }

    def build_score_stats_tables(self):
        if self.score_results is None:
            raise RuntimeError("NMI score data has not been computed")

        return {
            "nmi_score_pairwise_stats": pairwise_trace_stats(
                self.score_results["labels"].tolist(),
                self.score_results["traces"],
                self.discard,
            ),
        }

    def build_stats_tables(self):
        stats_tables = self.build_template_stats_tables()
        if self.score_results is not None:
            stats_tables.update(self.build_score_stats_tables())
        return stats_tables

    def save_report(self):
        self._require_out_dir()
        stats_tables = self.build_stats_tables()
        for stats_csv in save_stats_csvs(stats_tables, self.out_path).values():
            log.info(f"Saved stats table to: {stats_csv}")

        report_items = []
        if self.score_results is not None:
            df = pd.DataFrame(self.score_results["scores"].T, columns=self.labels)
            report_items.append(df.hvplot(width=1000))
            self._save_score_png(df)

        heatmaps = [
            pairwise_value_heatmap(
                stats_tables["nmi_template_pairwise_stats"],
                self.labels,
                "template_a",
                "template_b",
                "spatial_pearson_r",
                "Spatial Correlation Between Templates",
            )
        ]
        if self.score_results is not None:
            heatmaps.append(
                pairwise_value_heatmap(
                    stats_tables["nmi_score_pairwise_stats"],
                    self.labels,
                    "trace_a",
                    "trace_b",
                    "pearson_r",
                    "Temporal Correlation Between Score Traces",
                )
            )
        report_items.append(pn.Row(*heatmaps))

        report_suffix = "nmi_scores" if self.score_results is not None else "nmi_template_stats"
        html_out = f"{self.out_path}.{report_suffix}.html"
        pn.Column(*report_items).save(html_out)
        log.info(f"Saved NMI report to: {html_out}")
        return html_out

    def _save_score_png(self, df):
        fig = plt.figure(figsize=(20, 5))
        plt.plot(df)
        plt.xlabel("Time [TRs]")
        plt.ylabel("Signed NMI score")
        plt.legend(self.labels)
        png_out = f"{self.out_path}.nmi_scores.png"
        plt.savefig(png_out, dpi=200)
        plt.close(fig)
        log.info(f"Saved NMI score figure to: {png_out}")
        return png_out

    def run(self):
        self.build_template_data()
        self.save_template_data()
        if self.data_path is not None:
            self.score_data()
            self.save_score_data()
        return self.save_report()

    def _require_template_data(self):
        if self.template_data is None:
            if self.templates_path is None or self.mask_path is None:
                raise RuntimeError("NMI template data has not been built")
            self.build_template_data()

    def _require_out_dir(self):
        if not osp.isdir(self.out_dir):
            raise FileNotFoundError(f"Out directory does not exist: {self.out_dir}")


def process_options():
    parser = argparse.ArgumentParser(description="Prepare template maps and optionally score data for NMI matching")
    parser_inopts = parser.add_argument_group("Input Options", "Inputs to this program")
    parser_inopts.add_argument(
        "-d", "--data",
        action="store",
        type=file_exists,
        dest="data_path",
        default=None,
        help="Optional processed 4D data to score offline",
    )
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
    parser_inopts.add_argument(
        "--discard",
        action="store",
        type=int,
        dest="nvols_discard",
        default=100,
        help="Number of volumes to leave unscored at the beginning [Default: %(default)s]",
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
    OfflineNMI(process_options()).run()
