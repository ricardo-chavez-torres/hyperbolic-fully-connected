"""Tabulates a training-time benchmark: one runtime.json record per run, all from one GPU.

Written by runtime_exp/benchmark_training_time.sh. Each record is named
<model>_<eager|compiled>.json (the data-loader-only record is loader_only.json); runs that
were requested but failed are listed from the STATUS.txt next to them. The per-epoch cost of a run is the median
training time of its epochs after the first, which excludes CUDA warm-up and torch.compile's
compilation; the ratios divide it by the Euclidean ResNet-18's in the same mode, and by the
eager Euclidean ResNet-18.

Usage: summarize_training_benchmark.py RUN_DIR
"""
import json
import statistics
import sys
from pathlib import Path

ROWS = [
    ("loader_only", "data loader only"),
    ("euclidean", "Euclidean ResNet-18"),
    ("fgglnn", "FGG-LNN ResNet-18"),
    ("hcnn", "HCNN ResNet-18"),
    ("ilnn", "ILNN ResNet-18"),
    ("poincare", "Poincaré ResNet-20"),
]


def steady_epoch_time(record):
    times = [e["train_time_s"] for e in record.get("epochs", [])]
    if not times:
        return None
    return statistics.median(times[1:] if len(times) > 1 else times)


def main(run_dir):
    run_dir = Path(run_dir)
    records = {p.stem: json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))}
    if not records:
        print(f"No runtime records in {run_dir}")
        return 1

    # A run that crashed before its first epoch leaves a partial record; judge only real ones.
    finished = [r for r in records.values() if r.get("epochs")]
    gpus = sorted({r.get("gpu_name") or "?" for r in finished})
    iters = sorted({r.get("iterations_per_epoch") or 0 for r in finished})
    lines = [f"GPU: {', '.join(map(str, gpus))}   iterations/epoch: {', '.join(map(str, iters))}"]
    if len(gpus) > 1 or len(iters) > 1:
        lines.append("WARNING: records differ in GPU or iterations per epoch; ratios are not comparable.")
    print("\n".join(lines))

    base = {mode: steady_epoch_time(records[f"euclidean_{mode}"])
            for mode in ("eager", "compiled") if f"euclidean_{mode}" in records}
    base = {mode: t for mode, t in base.items() if t}

    # STATUS.txt (written by benchmark_training_time.sh) tells a failed run from one not requested.
    status = {}
    status_file = run_dir / "STATUS.txt"
    if status_file.exists():
        for entry in status_file.read_text().split("\n"):
            if "exit=" in entry:
                name, code = entry.split()[0], entry.rsplit("exit=", 1)[1].strip()
                status[name] = code

    header = (f"{'model':<22} {'mode':<9} {'epochs':>6} {'1st epoch s':>11} {'epoch s':>8} "
              f"{'x Eucl (same mode)':>18} {'x Eucl (eager)':>14}")
    print(header)
    print("-" * len(header))
    lines.append(header)
    for key, label in ROWS:
        modes = [None] if key == "loader_only" else ["eager", "compiled"]
        for mode in modes:
            stem = key if mode is None else f"{key}_{mode}"
            for suffix in ("", "_end"):
                record = records.get(stem + suffix)
                if record is not None and not record.get("epochs"):
                    record = None  # crashed before completing an epoch
                if record is None:
                    code = status.get(stem + suffix)
                    if code is not None:  # requested but no usable record
                        reason = f"failed (exit {code}), see {stem + suffix}.log" if code != "0" else "no record"
                        line = f"{label:<22} {mode or '':<9} {reason}"
                        print(line)
                        lines.append(line)
                    continue
                t = steady_epoch_time(record)
                first = record["epochs"][0]["train_time_s"] if record.get("epochs") else None
                same = t / base[mode] if mode in base and t else None
                eager = t / base["eager"] if "eager" in base and t else None
                line = (f"{label + (' (repeat)' if suffix else ''):<22} {mode or '':<9} "
                        f"{len(record.get('epochs', [])):>6} "
                        f"{first if first is not None else float('nan'):>11.1f} "
                        f"{t if t is not None else float('nan'):>8.2f} "
                        f"{(f'{same:.2f}x' if same else ''):>18} "
                        f"{(f'{eager:.2f}x' if eager and key != 'loader_only' else ''):>14}")
                print(line)
                lines.append(line)
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
