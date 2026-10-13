"""Package only source code and requested evidence; never datasets or weights."""
import zipfile
from pathlib import Path


def package_project(root, output, include_report=False):
    root, output = Path(root), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    directories = ["configs", "scripts", "tests", "lab_vlm", "student_checks", "student_tools"]
    if include_report:
        directories.append("report")
    filenames = ["README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md", "COLAB_GUIDE.md", "LIGHTNING_GUIDE.md",
                 "requirements-colab.txt", "requirements-export.txt", "pytest.ini", ".gitignore",
                 "lab_day10.ipynb", "lab_day10_lightning.ipynb"]
    paths = [root/name for name in filenames]
    for directory in directories:
        paths.extend((root/directory).rglob("*"))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in (".pyc", ".pt", ".bin", ".safetensors", ".ckpt"):
                continue
            relative = path.relative_to(root)
            if path.suffix in (".py", ".md", ".yaml", ".txt", ".ini", ".ipynb", ".json", ".png", ".jpg") or path.name == ".gitignore":
                archive.write(path, relative.as_posix())
    return output


def package_lightning_project(root, results_dir, output):
    """Recovery bundle: source/report plus selected original prediction and run JSONs.

    Inputs are in ignored work/; the submission ZIP created by package_project
    always omits work/. No dataset images, checkpoints or model weights are copied.
    """
    results_dir = Path(results_dir)
    output = package_project(root, output, include_report=True)
    paths = []
    for slug in ("e0", "e1", "e2", "e3"):
        paths.extend([results_dir/"work"/f"{slug}_test.jsonl",
                      results_dir/"work"/f"{slug}_test.meta.json"])
        if slug != "e0":
            paths.extend(results_dir/"work"/slug/name for name in ("run.json", "config.json", "target_quality.json"))
    paths.extend(results_dir/"work"/name for name in ("model_revision.json", "e4_selection.json"))
    with zipfile.ZipFile(output, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                archive.write(path, path.relative_to(results_dir).as_posix())
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(package_project(root, root/"work/colab-ready.zip"))
