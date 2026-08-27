"""Prepare templates and optionally evaluate data for Pearson matching."""

import logging
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from rtcog.matching.offline.template_utils import (
    load_masked_timeseries,
    prepare_template_data,
    score_pearson_timeseries,
    spatial_template_parser,
)


log = logging.getLogger("offline_pearson")


class OfflinePearson:
    """Run offline template preparation and evaluation for Pearson matching."""

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
        prefix="pearson",
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
        self.scores = None

    @property
    def labels(self):
        if self.template_data is None:
            raise RuntimeError("Pearson template data has not been prepared")
        return self.template_data["labels"].tolist()

    def build_template_data(self):
        self.template_data = prepare_template_data(
            self.templates_path,
            self.mask_path,
            self.template_labels_path,
        )
        return self.template_data

    def save_template_data(self):
        self._require_out_dir()
        if self.template_data is None:
            self.build_template_data()

        out_path = f"{self.out_path}.pearson_templates.npz"
        np.savez(out_path, **self.template_data)
        log.info("Saved Pearson template data to: %s", out_path)
        return out_path

    def score_data(self):
        if self.data_path is None:
            raise RuntimeError("Processed data were not provided for offline scoring")
        if self.template_data is None:
            self.build_template_data()

        data = load_masked_timeseries(self.data_path, self.mask_path)
        self.scores = score_pearson_timeseries(
            self.template_data,
            data,
            discard=self.discard,
        )
        return self.scores

    def save_score_data(self):
        self._require_out_dir()
        if self.scores is None:
            self.score_data()

        score_path = f"{self.out_path}.pearson_scores.npy"
        trace_path = f"{self.out_path}.pearson_score_traces.npz"
        np.save(score_path, self.scores)
        np.savez(
            trace_path,
            **{label: self.scores[index] for index, label in enumerate(self.labels)},
        )
        log.info("Saved Pearson scores to: %s", score_path)
        log.info("Saved Pearson score traces to: %s", trace_path)
        return {"scores": score_path, "traces": trace_path}

    def save_score_plot(self):
        """Save a quick-look plot of the offline Pearson score traces."""
        self._require_out_dir()
        if self.scores is None:
            self.score_data()

        figure = plt.figure(figsize=(20, 5))
        plt.plot(self.scores.T)
        plt.xlabel("Time [TRs]")
        plt.ylabel("Pearson r")
        plt.ylim(-1, 1)
        plt.legend(self.labels)
        plot_path = f"{self.out_path}.pearson_scores.png"
        plt.savefig(plot_path, dpi=200)
        plt.close(figure)
        log.info("Saved Pearson score plot to: %s", plot_path)
        return plot_path

    def run(self):
        self.build_template_data()
        outputs = {"templates": self.save_template_data()}
        if self.data_path is not None:
            self.score_data()
            outputs.update(self.save_score_data())
            outputs["plot"] = self.save_score_plot()
        return outputs

    def _require_out_dir(self):
        if not osp.isdir(self.out_dir):
            raise FileNotFoundError(f"Out directory does not exist: {self.out_dir}")


def process_options(argv=None):
    parser, _, _ = spatial_template_parser("Pearson", "pearson")
    return parser.parse_args(argv)


def main(argv=None):
    return OfflinePearson(process_options(argv)).run()


if __name__ == "__main__":
    main()
