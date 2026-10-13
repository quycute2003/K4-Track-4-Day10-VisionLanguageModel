"""Finish/export saved experiments on CPU, independent of Colab and GPU runtimes."""
import hashlib
import json
import shutil
from pathlib import Path

from scripts.labcommon import read_jsonl
from .data import write_json
from .packaging import package_project
from .reporting import make_summary, prediction_figures

PUBLIC_DATASET = "https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"


def _copy_original(source, target):
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError(f"Conflicting existing result: {target}; choose a fresh destination")
    shutil.copy2(source, target)


def finish_saved_results(repo_dir, results_dir=None, data_dir=None, output_dir=None, fetch_data=True):
    """Restore original evidence, make figures/table, and package source + results.

    No training/inference is called. Raw predictions live under results_dir/work/;
    all artifacts written for submission exclude data, predictions and weights.
    """
    root = Path(repo_dir).resolve()
    source = Path(results_dir or root).resolve()
    data = Path(data_dir or root/"data").resolve()
    outputs = Path(output_dir or root/"exports").resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for slug in ("e0", "e1", "e2", "e3"):
        path = source/"report/evidence"/f"{slug}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing saved evidence {path}; resume scoring/inference with the saved adapter, not training")
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("score_version") != 1 or d.get("split") != "test" or d.get("missing_predictions") != 0:
            raise ValueError(f"Incomplete/non-test evidence: {path}")
        if d.get("coord") != "norm1000":
            raise ValueError(f"Expected baseline norm1000 evidence: {path}")
        evidence[slug] = d
    versions = {d["dataset_version"] for d in evidence.values()}
    if len(versions) != 1:
        raise ValueError("Saved experiments use different dataset versions")
    dataset_version = next(iter(versions))

    for slug, d in evidence.items():
        _copy_original(source/"report/evidence"/f"{slug}.json", root/"report/evidence"/f"{slug}.json")
    # Restore other report artifacts, preserving any destination files already edited.
    for path in (source/"report").rglob("*"):
        if path.is_file():
            target = root/"report"/path.relative_to(source/"report")
            if not target.exists():
                _copy_original(path, target)
    if not (root/"REPORT.md").exists() and (source/"REPORT.md").exists():
        _copy_original(source/"REPORT.md", root/"REPORT.md")
    if not (root/"REPORT.md").exists():
        raise FileNotFoundError("REPORT.md template is missing from the source package")

    candidates = []
    validation = {}
    for slug in ("e1", "e2", "e3"):
        val_path = source/"report/validation"/f"{slug}.json"
        if val_path.exists():
            val = json.loads(val_path.read_text(encoding="utf-8"))
            if val.get("split") != "val" or val.get("dataset_version") != dataset_version:
                raise ValueError(f"Wrong validation evidence: {val_path}")
            validation[slug] = val["micro"]["f1"]
        pred = source/"work"/f"{slug}_test.jsonl"
        if pred.exists():
            raw_rows = read_jsonl(pred)
            if len(raw_rows) != evidence[slug]["n_images"] or len({r["image_id"] for r in raw_rows}) != len(raw_rows):
                raise ValueError(f"Incomplete/duplicate saved predictions: {pred}")
            expected_hash = evidence[slug].get("meta", {}).get("prediction_sha256")
            if expected_hash and hashlib.sha256(pred.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Saved prediction hash differs from evidence: {pred}")
            candidates.append(slug)
        run = source/"work"/slug/"run.json"
        if run.exists():
            _copy_original(run, root/"report/runs"/f"{slug}.json")
        quality = source/"work"/slug/"target_quality.json"
        if quality.exists():
            _copy_original(quality, root/"report/diagnostics"/f"{slug}_target_quality.json")

    existing_figures = len(list((root/"report/figures").glob("prediction_*.png"))) >= 6
    if not candidates and not existing_figures:
        raise FileNotFoundError("Need at least one complete e1/e2/e3_test.jsonl to produce prediction figures; weights are not needed")
    eligible = [slug for slug in candidates if slug in validation]
    best = max(eligible, key=validation.get) if eligible else candidates[0] if candidates else None
    criterion = "validation micro F1" if eligible else "first available saved prediction file"
    print("Saved test evidence E0–E3 recovered; dataset:", dataset_version, flush=True)
    print("Figure model:", best or "existing figures", "criterion:", criterion, flush=True)

    e0_pred = source/"work/e0_test.jsonl"
    coord_path = root/"report/diagnostics/e0_wrong_coord.json"
    need_data = not existing_figures or (e0_pred.exists() and not coord_path.exists())
    meta_path = data/"splits/meta.json"
    if need_data and not meta_path.exists():
        if not fetch_data:
            raise FileNotFoundError("Need dataset images/splits for new figures. Allow public dataset download or supply data_dir.")
        from scripts.prepare_data import main as prepare_data
        print("Downloading public dataset for CPU figure rendering; no model is loaded...", flush=True)
        prepare_data(["--zip", PUBLIC_DATASET, "--out", str(data), "--work", str(root/"work/export-dataset")])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["dataset_version"] != dataset_version:
            raise ValueError("Dataset differs from the saved experiment evidence")
    if not existing_figures:
        prediction_figures(source/"work"/f"{best}_test.jsonl", data, root/"report/figures")
        print("Rendered 6 prediction/ground-truth figures.", flush=True)
    if e0_pred.exists() and not coord_path.exists():
        expected = evidence["e0"].get("meta", {}).get("prediction_sha256")
        if expected and hashlib.sha256(e0_pred.read_bytes()).hexdigest() != expected:
            raise ValueError("E0 predictions differ from saved evidence")
        from scripts.score import main as score
        score(["--pred", str(e0_pred), "--name", "E0_WRONG_COORD", "--coord", "pixel",
               "--out", str(coord_path), "--splits-dir", str(data/"splits")])
    make_summary(root/"report/evidence", root/"report/SUMMARY.md", root/"REPORT.md")
    write_json(root/"report/diagnostics/export_state.json", {
        "dataset_version": dataset_version, "figure_model": best, "selection_criterion": criterion,
        "export_from_saved_results": True, "training_or_inference_performed": False,
        "e4_evidence_present": any((root/"report/evidence").glob("e4*.json")),
        "e5_evidence_present": (root/"report/evidence/e5.json").exists(),
    })
    export = package_project(root, outputs/"day10-results.zip", include_report=True)
    print("Exported:", export, flush=True)
    print("Review REPORT.md before submitting. Missing E4/E5 evidence is not fabricated.", flush=True)
    return export
