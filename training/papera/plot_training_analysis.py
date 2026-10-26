"""Render auditable, offline-only ACT training figures; never loads a checkpoint.

Use the existing mujoco_repro Python (Matplotlib is already installed). Model
evaluation remains in paperA_train. Source metrics and evaluation files are read
only; exports and exact plotting tables live in RUN/analysis/figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BLUE = "#0072B2"
ORANGE = "#B34C00"
GREEN = "#007B65"
GRAY = "#606060"
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "figure.titlesize": 13, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, directory, stem, description, manifest):
    for extension in ("png", "pdf"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white", transparent=False)
        record = {"path": path.name, "sha256": sha256(path),
                  "width_inches": fig.get_figwidth(), "height_inches": fig.get_figheight(),
                  "description": description}
        if extension == "png":
            with Image.open(path) as image:
                record.update(pixels=list(image.size), dpi=list(image.info.get("dpi", [])),
                              mode=image.mode)
        manifest["artifacts"].append(record)
    plt.close(fig)


def line(ax, rows, key, *, color=BLUE, label=None, marker=None, linewidth=1.3):
    ax.plot([row["step"] for row in rows], [row.get(key, np.nan) for row in rows],
            color=color, label=label, marker=marker, markersize=3, linewidth=linewidth)
    ax.set_xlabel("Optimizer step")
    ax.set_xlim(left=0)


def analysis_context(run, evaluation, *, required=False):
    """Bind optional offline results to this run and label their actual scope."""
    run_info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    provenance = run_info["provenance"]
    scope = provenance.get("validation_scope", "overfit_reconstruction"
                           if provenance.get("engineering_overfit") else "heldout_sessions")
    if scope not in ("heldout_sessions", "overfit_reconstruction"):
        raise ValueError(f"Unsupported validation scope: {scope}")
    summary_path = evaluation / "summary.json"
    episodes = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (not summary.get("complete") or summary.get("signature") != run_info["signature"]
                or Path(summary.get("run", "")).resolve() != run.resolve()):
            raise ValueError("Analysis summary does not belong to this completed run analysis")
        if summary.get("validation_scope", scope) != scope:
            raise ValueError("Analysis validation scope differs from the run")
        results = [row for row in summary["results"]
                   if row["checkpoint"] == "best" and row["split"] == "val"]
        if len(results) != 1 or type(results[0].get("episodes")) is not int or results[0]["episodes"] < 1:
            raise ValueError("Analysis summary needs one best-val result with an episode count")
        episodes = results[0]["episodes"]
    elif required:
        raise FileNotFoundError(f"Completed analysis summary is missing: {summary_path}")
    reconstruction = scope == "overfit_reconstruction"
    return {"scope": scope, "episodes": episodes,
            "label": "Reconstruction" if reconstruction else "Validation",
            "population": "training-subset reconstruction" if reconstruction else "held-out evaluation",
            "episode_label": "reconstruction" if reconstruction else "held-out",
            "generalization_evidence": not reconstruction}


def log_figures(run, directory, manifest, context=None, max_step=None):
    metrics_path = run / "metrics.jsonl"
    rows = [json.loads(row) for row in metrics_path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if max_step is not None:
        if type(max_step) is not int or max_step < 1:
            raise ValueError("max_step must be a positive integer")
        rows = [row for row in rows if row.get("step", 0) <= max_step]
    train = [row for row in rows if row["event"] == "train"]
    validation = [row for row in rows if row["event"] == "validation"]
    if not train or not validation:
        raise ValueError("Both training and validation log records are required.")
    best = min(validation, key=lambda row: row["prior_action_l1"])
    run_info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    config = run_info["provenance"]["config"]
    variant = run_info["provenance"]["variant"]
    context = context or analysis_context(run, run / "analysis" / "evaluation")
    label = context["label"]
    write_csv(directory / "training_log_points.csv", train)
    write_csv(directory / "validation_log_points.csv", validation)
    for filename in ("metrics.jsonl", "run.json", "completion.json"):
        path = run / filename
        if path.exists():
            manifest["sources"].append({"path": str(path), "sha256": sha256(path)})
    manifest["log_figures"] = {"train_points": len(train), "validation_points": len(validation),
        "best_selection": "minimum logged validation prior_action_l1", "best_step": best["step"],
        "max_step": max_step,
        "validation_scope": context["scope"], "validation_episodes": context["episodes"],
        "smoothing": "none", "error_bars": "none: one run, overlapping windows are not independent replicates",
        "zero_log_policy": "nonpositive values retained in CSV and shown as gaps on logarithmic axes"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout="constrained")
    suffix = f", through step {max_step:,}" if max_step is not None else ""
    fig.suptitle(f"ACT {variant}: offline learning curves (one seed{suffix})")
    line(axes[0, 0], train, "action", label="Train: posterior, one logged minibatch", linewidth=.6)
    line(axes[0, 0], validation, "prior_action_l1", color=ORANGE,
         label=f"{label}: prior z=0, {context['population']}", marker="o")
    axes[0, 0].set(title="A  Normalized action L1: different protocols", ylabel="Mean absolute z-score error")
    axes[0, 0].legend(loc="upper right")
    fields = [(axes[0, 1], "xyz_error_mm", f"B  {label} XYZ", "Mean command XYZ Euclidean error (mm)"),
              (axes[1, 0], "gripper_error_mm", f"C  {label} gripper width", "Mean absolute width error (mm)"),
              (axes[1, 1], "rpy_mae_degrees", f"D  {label} Euler angles", "Wrapped per-axis mean error (degrees)")]
    for ax, key, title, ylabel in fields:
        line(ax, validation, key, color=ORANGE, marker="o")
        ax.set(title=title, ylabel=ylabel, ylim=(0, None))
        ax.axvline(best["step"], color=GRAY, linestyle="--", linewidth=.8,
                   label=f"Best L1 checkpoint: {best['step']:,}")
        ax.legend(loc="upper right")
    caveat = "Same training episodes; not generalization." if context["scope"] == "overfit_reconstruction" else "Not task success."
    fig.supxlabel(f"No smoothing. {label} pools valid targets over {config['model']['chunk_size']}-step chunks. {caveat}", fontsize=9)
    save_figure(fig, directory, "01_learning_curves",
        f"Raw logged training action L1 and {context['population']} prior-only action L1, XYZ, gripper and wrapped Euler errors. "
        "Training uses a sampled posterior and dropout; its minibatch values are not a matched generalization comparison. "
        "Dashed lines identify the checkpoint chosen by lowest validation normalized action L1.", manifest)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout="constrained")
    fig.suptitle(f"ACT {variant}: optimization diagnostics (raw logged minibatches)")
    for ax, key, title, ylabel in [
        (axes[0, 0], "total", "A  Training objective", "Total weighted objective (log scale)"),
        (axes[0, 1], "kl", "B  CVAE KL term", f"KL: sum of {config['model']['latent_dim']} latent dimensions (log scale)"),
        (axes[1, 0], "gradient_norm", "C  Gradient norm before clipping", "Global gradient norm (log scale)"),
    ]:
        shown = [dict(row, **{key: row[key] if row[key] > 0 else np.nan}) for row in train]
        line(ax, shown, key, linewidth=.75)
        ax.set(title=title, ylabel=ylabel, yscale="log")
    axes[1, 0].axhline(config["gradient_clip_norm"], color=ORANGE, linestyle="--",
                      label=f"Clip limit = {config['gradient_clip_norm']:g}")
    axes[1, 0].legend()
    line(axes[1, 1], train, "lr", color=GREEN)
    axes[1, 1].set(title="D  Main learning rate", ylabel="Learning rate after scheduler update", ylim=(0, None))
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.supxlabel(f"One record every {config['log_every']} steps plus step 1; unsaved minibatches cannot be diagnosed from this log.", fontsize=9)
    save_figure(fig, directory, "02_optimization_diagnostics",
        "Raw training objective, latent KL, pre-clipping global gradient norm, and main learning rate. "
        "First three panels use logarithmic y scales. Large pre-clipping values are not unclipped updates. "
        "KL near zero alone cannot establish whether useful latent information is retained.", manifest)


def filter_metrics(rows, **criteria):
    return [row for row in rows if all(str(row.get(key)) == str(value) for key, value in criteria.items())]


def evaluation_figures(run, directory, manifest, evaluation=None, context=None):
    evaluation = evaluation or run / "analysis" / "evaluation"
    required = [evaluation / f"{name}.json" for name in ("best_val", "best_train", "last_val")]
    if not all(path.is_file() for path in required):
        manifest["evaluation_figures"] = "not available: need best_val.json, best_train.json and last_val.json"
        return
    context = context or analysis_context(run, evaluation, required=True)
    loaded = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in required}
    for result in loaded.values():
        if result.get("validation_scope", context["scope"]) != context["scope"]:
            raise ValueError("Analysis result validation scope differs from its summary")
    tables = {name: value["metrics"] for name, value in loaded.items()}
    for path in [evaluation / "summary.json", *required]:
        manifest["sources"].append({"path": str(path), "sha256": sha256(path)})
    val = tables["best_val"]
    episode = sorted(filter_metrics(val, group="episode", predictor="model", horizon="first", arm="both"),
                     key=lambda row: (row["xyz_error_mm_mean"], row["value"]))
    if not episode or len(episode) != context["episodes"] or loaded["best_val"]["episodes"] != len(episode):
        raise ValueError("Analysis episode rows do not match the completed summary")
    aliases = {row["value"]: f"E{index+1:02d}" for index, row in enumerate(episode)}
    grip_order = sorted(episode, key=lambda row: (row["gripper_error_mm_mean"], row["value"]))
    selection = [("Median gripper error", grip_order[len(grip_order)//2]), ("Largest gripper error", grip_order[-1])]
    manifest["episode_aliases"] = aliases
    manifest["evaluation_figures"] = {
        "episode_order": "ascending best-val first-target both-arm XYZ mean; lexical key breaks ties",
        "trace_selection": [{"selection": label, "episode": row["value"], "alias": aliases[row["value"]],
                             "gripper_error_mm_mean": row["gripper_error_mm_mean"]} for label, row in selection],
        "trace_selection_metric": "best-val first-target both-arm mean absolute gripper width error; lexical key breaks ties",
        "median_selection": f"upper middle ordered episode for even n; all n={len(episode)} {context['episode_label']} episodes included in breakdown",
        "validation_scope": context["scope"], "validation_episodes": len(episode),
        "stage_semantics": "stage at observation anchor, not necessarily future target stage",
        "horizon_support": "valid window count and composition vary with horizon because padded tail targets are excluded; exact counts in CSV",
        "intervals": "stage p95 is empirical arm-target error percentile, not confidence interval",
    }
    summary = []
    for name in ("best_train", "best_val", "last_val"):
        rows = tables[name]
        row = filter_metrics(rows, group="all", predictor="model", horizon="all", arm="both")[0]
        summary.append(dict(row, evaluation=name))
    source_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    fig.suptitle(f"Checkpoint analysis: {context['population']}, prior z=0; accepted command targets")
    x = np.arange(len(summary))
    labels = ["Best / train", f"Best / {context['label'].lower()}", f"Last / {context['label'].lower()}"]
    axes[0, 0].plot(x, [row["normalized_action_l1_mean"] for row in summary],
                    linestyle="none", marker="o", color=BLUE, markersize=7)
    for pos, row in zip(x, summary):
        axes[0, 0].annotate(f"{row['normalized_action_l1_mean']:.4f}\nn={row['windows']:,}",
                           (pos, row["normalized_action_l1_mean"]), xytext=(0, 10),
                           textcoords="offset points", ha="center", fontsize=8)
    axes[0, 0].set(xticks=x, xticklabels=labels, ylabel="Normalized action L1", title="A  Matched prior-only comparison",
                    ylim=(0, max(row["normalized_action_l1_mean"] for row in summary)*1.45), xlim=(-.5, 2.5))
    axes[0, 0].tick_params(axis="x", labelsize=8)
    source_rows.extend(summary)

    for ax, field, title, ylabel in [
        (axes[0, 1], "xyz_error_mm_mean", f"B  {context['label']} by prediction horizon", "Mean command XYZ error (mm)"),
        (axes[0, 2], "gripper_error_mm_mean", f"C  {context['label']} by prediction horizon", "Mean gripper target error (mm)"),
    ]:
        for arm, color, marker in (("left", BLUE, "o"), ("right", ORANGE, "s")):
            rows = sorted(filter_metrics(val, group="horizon", predictor="model", arm=arm), key=lambda row:int(row["horizon"]))
            ax.plot([int(row["horizon"]) for row in rows], [row[field] for row in rows],
                     color=color, marker=marker, markersize=3, label=f"{arm.capitalize()} arm")
            source_rows.extend(rows)
        ax.set(title=title, ylabel=ylabel, xlabel="Target index (0 = next control tick)", ylim=(0, None))
        ax.legend()

    y = np.arange(len(episode))
    for arm, color, marker, offset in (("left", BLUE, "o", -.12), ("right", ORANGE, "s", .12)):
        rows = [filter_metrics(val, group="episode", value=row["value"], predictor="model", horizon="first", arm=arm)[0] for row in episode]
        axes[1, 0].plot([row["xyz_error_mm_mean"] for row in rows], y+offset,
                        color=color, marker=marker, markersize=4, linestyle="none", label=f"{arm.capitalize()} arm")
        source_rows.extend(rows)
    axes[1, 0].set(yticks=y, yticklabels=[f"{aliases[row['value']]} (n={row['valid_targets_per_arm']})" for row in episode],
                    title=f"D  Every {context['episode_label']} episode: first target", xlabel="Mean command XYZ error (mm)", xlim=(0, None))
    axes[1, 0].tick_params(axis="y", labelsize=8)
    axes[1, 0].legend(loc="best")
    stages = filter_metrics(val, group="anchor_stage", predictor="model", horizon="first", arm="both")
    source_rows.extend(stages)
    for ax, metric, title in [(axes[1, 1], "xyz_error_mm", "E  First-target error by anchor stage"),
                              (axes[1, 2], "gripper_error_mm", "F  First-target error by anchor stage")]:
        for stat, color, marker, label in (("mean", BLUE, "o", "Mean"), ("p95", ORANGE, "x", "95th percentile (not CI)")):
            ax.plot([row[metric+"_"+stat] for row in stages], np.arange(len(stages)),
                     linestyle="none", color=color, marker=marker, markersize=5, label=label)
        ax.set(title=title, yticks=np.arange(len(stages)),
               yticklabels=[f"{row['value'].replace('_', ' ')}\n(n={row['valid_targets_per_arm']})" for row in stages],
               xlabel=("Command XYZ error (mm)" if metric == "xyz_error_mm" else "Gripper target error (mm)"), xlim=(0, None))
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(loc="lower right", fontsize=7)
    fig.supxlabel("n = valid windows / first targets; stages without samples are absent. Horizon supports differ due to padding (counts in CSV).\nEpisode keys and exact values are in CSV/manifest. These are command prediction errors, not task success.", fontsize=9)
    unique_rows = {}
    for row in source_rows:
        key = tuple(str(row.get(field, "")) for field in
                    ("evaluation", "checkpoint", "split", "predictor", "group", "value", "horizon", "arm"))
        unique_rows[key] = row
    write_csv(directory / "checkpoint_breakdown_points.csv", list(unique_rows.values()))
    save_figure(fig, directory, "03_checkpoint_breakdown",
        f"Matched prior-only train and {context['population']}, left/right future-horizon errors, all {context['episode_label']} episode first-target errors, "
        "and anchor-stage mean/95th-percentile errors. Horizon counts and composition differ because padded tail targets are excluded. "
        "Error distributions pool overlapping command targets; percentiles are descriptive, not confidence intervals. "
        "All quantities compare model command predictions with demonstrated accepted commands, not actual robot pose or success.", manifest)

    trace_path = evaluation / "best_val_first_targets.npz"
    if not trace_path.is_file():
        return
    manifest["sources"].append({"path": str(trace_path), "sha256": sha256(trace_path)})
    trace_rows = []
    with np.load(trace_path, allow_pickle=False) as data:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained", sharey=True)
        fig.suptitle("First-target gripper commands: demonstrated reference vs offline prediction")
        for ri, (label, row) in enumerate(selection):
            indices = np.flatnonzero(data["episode_key"] == row["value"])
            times = data["original_time_s"][indices]
            relative = times-times[0]
            breaks = np.r_[True, (np.diff(data["raw_frame_index"][indices]) != 1)
                | (np.diff(times) > .05) | (np.diff(times) <= 0)
                | (data["segment_id"][indices][1:] != data["segment_id"][indices][:-1])
                | (data["control_epoch"][indices][1:] != data["control_epoch"][indices][:-1])]
            segment = np.cumsum(breaks)
            valid = data["valid"][indices]
            for ai, (arm, off) in enumerate((("left", 6), ("right", 13))):
                ax = axes[ri, ai]
                predicted = data["prediction"][indices, off]*1000
                target = data["target"][indices, off]*1000
                for si in np.unique(segment):
                    mask = segment == si
                    pred_plot = np.where(valid[mask], predicted[mask], np.nan)
                    true_plot = np.where(valid[mask], target[mask], np.nan)
                    ax.plot(relative[mask], true_plot, color="#222222", linewidth=1.25,
                            label="Demonstrated target" if si == segment[0] else None)
                    ax.plot(relative[mask], pred_plot, color=ORANGE, linestyle="--", linewidth=1.0,
                            label="Predicted target" if si == segment[0] else None)
                for j, source_index in enumerate(indices):
                    trace_rows.append({"selection": label, "episode_alias": aliases[row["value"]], "episode_key": row["value"],
                        "arm": arm, "raw_frame_index": int(data["raw_frame_index"][source_index]),
                        "original_time_s": float(times[j]), "relative_time_s": float(relative[j]),
                        "segment_id": str(data["segment_id"][source_index]), "control_epoch": str(data["control_epoch"][source_index]),
                        "plot_segment": int(segment[j]), "valid": bool(valid[j]),
                        "demonstrated_target_width_mm": float(target[j]) if valid[j] else None,
                        "predicted_target_width_mm": float(predicted[j]) if valid[j] else None})
                ax.set(title=f"{label}: {aliases[row['value']]} / {arm} gripper", xlabel="Original elapsed episode time (s)", ylabel="Target total width (mm)")
                ax.legend(loc="best")
        fig.supxlabel("Episodes selected by median / largest both-arm first-target gripper width error. Lines break at segment, epoch, frame or time gaps.\nTargets for the next control tick are plotted at observation time; these are not measured gripper widths or rollouts.", fontsize=9)
        write_csv(directory / "selected_first_target_traces.csv", trace_rows)
        save_figure(fig, directory, "04_first_target_gripper_traces",
            f"Both gripper first-target command traces for the median and highest mean gripper-width-error {context['episode_label']} episodes, selected using all {len(episode)} episodes. "
            "Black solid lines are demonstrated accepted target commands; orange dashed lines are offline predictions. "
            "Original timing is preserved and discontinuities are left blank. These are not physical executed trajectories.", manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path,
                        help="Completed analyze_checkpoint.py --output directory; defaults to RUN/analysis/evaluation")
    parser.add_argument("--output", type=Path,
                        help="Figure output directory; defaults to RUN/analysis/figures")
    parser.add_argument("--max-step", type=int,
                        help="Plot only logged training/validation rows at or below this optimizer step")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    evaluation = args.analysis_dir.resolve() if args.analysis_dir else run / "analysis" / "evaluation"
    context = analysis_context(run, evaluation, required=args.analysis_dir is not None)
    directory = args.output.resolve() if args.output else run / "analysis" / "figures"
    if directory == run or directory in run.parents or directory == evaluation:
        raise ValueError("Figure output must be separate from the run root and analysis inputs")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "papera-analysis-figures-v1", "run": str(run),
        "analysis_directory": str(evaluation), "validation_context": context,
        "purpose": "Exploratory offline research analysis; no target journal specified",
        "publisher_requirements": "not evaluated: no submission target specified",
        "software": {"python": platform.python_version(), "executable": sys.executable,
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "source_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "sources": [], "artifacts": [], "table_policy": "CSV exports contain exact plotted source values; no smoothing",
        "uncertainty_policy": "No confidence intervals: one seed; windows overlap within sessions",
        "skill_reference": "https://arxiv.org/abs/2609.00065"}
    with plt.rc_context(STYLE):
        log_figures(run, directory, manifest, context, max_step=args.max_step)
        evaluation_figures(run, directory, manifest, evaluation, context)
    for path in directory.glob("*.csv"):
        manifest["artifacts"].append({"path": path.name, "sha256": sha256(path), "type": "source table"})
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(directory), "artifacts": len(manifest["artifacts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
