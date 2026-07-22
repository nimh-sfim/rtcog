import numpy as np
import pandas as pd
import hvplot.pandas


def _template_matrix(templates):
    templates = np.asarray(templates, dtype=np.float64)
    if templates.ndim == 1:
        return templates[np.newaxis, :]
    if templates.ndim != 2:
        raise ValueError("templates must be a 1D or 2D array")
    return templates


def _mask_matrix(masks, templates):
    if masks is None:
        return templates != 0

    masks = np.asarray(masks, dtype=bool)
    if masks.ndim == 1:
        return masks[np.newaxis, :]
    if masks.ndim != 2:
        raise ValueError("masks must be a 1D or 2D array")
    return masks


def _safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan

    x = x[finite]
    y = y[finite]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_template_stats(labels, templates, masks=None):
    templates = _template_matrix(templates)
    masks = _mask_matrix(masks, templates)

    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            mask_i = masks[i]
            mask_j = masks[j]

            selected_intersection = int(np.logical_and(mask_i, mask_j).sum())

            rows.append({
                "template_a": str(labels[i]),
                "template_b": str(labels[j]),
                "overlap_voxels": selected_intersection,
                "spatial_pearson_r": _safe_pearson(templates[i], templates[j]),
            })
    return pd.DataFrame(rows)


def pairwise_trace_stats(labels, traces, discard=0):
    rows = []
    raw_traces = {}
    for label in labels:
        trace = np.asarray(traces[label], dtype=np.float64).ravel()[discard:]
        raw_traces[label] = trace

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_i = labels[i]
            label_j = labels[j]
            raw_i = raw_traces[label_i]
            raw_j = raw_traces[label_j]

            rows.append({
                "trace_a": str(label_i),
                "trace_b": str(label_j),
                "pearson_r": _safe_pearson(raw_i, raw_j),
            })
    return pd.DataFrame(rows)


def pairwise_value_matrix(table, labels, label_a_col, label_b_col, value_col, diagonal=1.0):
    labels = [str(label) for label in labels]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=np.float64)
    np.fill_diagonal(matrix.values, diagonal)

    for _, row in table.iterrows():
        label_a = str(row[label_a_col])
        label_b = str(row[label_b_col])
        value = row[value_col]
        matrix.loc[label_a, label_b] = value
        matrix.loc[label_b, label_a] = value

    return matrix


def pairwise_value_long(table, labels, label_a_col, label_b_col, value_col, value_name="value", diagonal=1.0):
    matrix = pairwise_value_matrix(
        table,
        labels,
        label_a_col,
        label_b_col,
        value_col,
        diagonal=diagonal,
    )
    label_axis_name = label_a_col.rsplit("_", 1)[0] or "label"
    long = (
        matrix
        .rename_axis(label_axis_name)
        .reset_index()
        .melt(id_vars=label_axis_name, var_name=f"{label_axis_name}_comparison", value_name=value_name)
    )
    return long


def pairwise_value_heatmap(
    table,
    labels,
    label_a_col,
    label_b_col,
    value_col,
    title,
    value_name="Pearson r",
    diagonal=1.0,
    width=450,
    height=450,
):
    label_axis_name = label_a_col.rsplit("_", 1)[0] or "label"
    comparison_axis_name = f"{label_axis_name}_comparison"
    long = pairwise_value_long(
        table,
        labels,
        label_a_col,
        label_b_col,
        value_col,
        value_name=value_name,
        diagonal=diagonal,
    )
    return long.hvplot.heatmap(
        x=comparison_axis_name,
        y=label_axis_name,
        C=value_name,
        cmap="RdBu_r",
        clim=(-1, 1),
        colorbar=True,
        title=title,
        width=width,
        height=height,
    ).opts(
        xrotation=45,
        yaxis="left",
        tools=["hover"],
    )


def save_stats_csvs(tables, out_path):
    out_paths = {}
    for name, df in tables.items():
        filename = f"{out_path}.{name}.csv"
        df.to_csv(filename, index=False)
        out_paths[name] = filename
    return out_paths
