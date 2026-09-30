import re
import sys

import pandas as pd

def parse_results(path="results.txt"):
    with open(path) as f:
        text = f.read()

    blocks = re.split(r"={40,}", text)

    rows = []
    for i, block in enumerate(blocks):
        header = re.search(
            r"(\w+(?:_\w+)*)\s+(NC|LP)\s*\|\s*(\w+)\s+encoder\s*\+\s*(\w+)\s+decoder\s*\|\s*seed=(\d+)",
            block,
        )
        if not header:
            continue

        dataset, task, encoder, decoder, seed = header.groups()

        # Search this block + next for the metrics (they follow the header)
        search = block + (blocks[i + 1] if i + 1 < len(blocks) else "")

        lr = re.search(r"Learning rate:\s*([\d.]+)", search)
        curv = re.search(r"Curvature:\s*(\w+)", search)
        time_ = re.search(r"Total time elapsed:\s*([\d.]+)s", search)
        params = re.search(r"Total number of parameters:\s*([\d.]+)", search)

        # Test metrics
        test_line = re.search(r"Test set results:\s*(.*)", search)
        test_metrics = {}
        if test_line:
            for m in re.finditer(r"(test_\w+):\s*([\d.]+)", test_line.group(1)):
                test_metrics[m.group(1)] = float(m.group(2))

        # Val metrics
        val_line = re.search(r"Val set results:\s*(.*)", search)
        val_metrics = {}
        if val_line:
            for m in re.finditer(r"(val_\w+):\s*([\d.]+)", val_line.group(1)):
                val_metrics[m.group(1)] = float(m.group(2))

        row = {
            "dataset": dataset,
            "task": task,
            "encoder": encoder,
            "decoder": decoder,
            "seed": int(seed),
            "lr": float(lr.group(1)) if lr else None,
            "curvature": curv.group(1) if curv else None,
            "time_s": float(time_.group(1)) if time_ else None,
            "params": float(params.group(1)) if params else None,
            **val_metrics,
            **test_metrics,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(df):
    """Mean and std over seeds per dataset, task, encoder, learning rate and curvature.

    Returns (node classification, link prediction) tables: accuracy/F1 for NC,
    ROC-AUC/AP for LP.
    """
    metrics = [m for m in ("test_acc", "test_f1", "test_roc", "test_ap") if m in df.columns]
    aggregations = {"n": ("seed", "count")}
    for m in metrics:
        aggregations[f"{m}_mean"] = (m, "mean")
        aggregations[f"{m}_std"] = (m, "std")
    summary = (
        df.groupby(["dataset", "task", "encoder", "lr", "curvature"])
        .agg(**aggregations)
        .reset_index()
    )

    def keep(task, task_metrics):
        drop = [c for c in summary.columns
                if c.startswith("test_") and not c.rsplit("_", 1)[0] in task_metrics]
        return summary[summary["task"] == task].drop(columns=drop)

    return keep("NC", ("test_acc", "test_f1")), keep("LP", ("test_roc", "test_ap"))


if __name__ == "__main__":
    # Usage: python parse_results.py [RESULTS_FILE]   (train.py appends to results.txt)
    df = parse_results(sys.argv[1] if len(sys.argv) > 1 else "results.txt")
    print(f"{len(df)} experiments parsed")
    print("\nDataset x Task x Encoder counts:")
    print(df.groupby(["dataset", "task", "encoder"]).size().to_string())
    nc, lp = summarize(df)
    print("\n=== Node Classification ===")
    print(nc.to_string(index=False))
    print("\n=== Link Prediction ===")
    print(lp.to_string(index=False))
