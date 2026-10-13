"""Use `py -m lab_vlm.cli --help` on Windows; sys.executable on Colab."""
import argparse
import json
from pathlib import Path

import yaml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("train", "infer"):
        parser = sub.add_parser(command)
        parser.add_argument("--config", required=True)
        parser.add_argument("--data-dir", default="data")
        if command == "train":
            parser.add_argument("--run-dir", required=True)
        else:
            parser.add_argument("--out", required=True)
            parser.add_argument("--checkpoint")
            parser.add_argument("--split", default="test", choices=["val", "test"])
            parser.add_argument("--batch-size", type=int, default=4)
            parser.add_argument("--limit", type=int)
    parser = sub.add_parser("score")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--setup")
    parser = sub.add_parser("report")
    parser.add_argument("--evidence-dir", default="report/evidence")
    parser.add_argument("--out", default="report/SUMMARY.md")
    parser.add_argument("--fill-table", action="store_true")
    parser = sub.add_parser("figures")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="report/figures")
    parser.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    parser = sub.add_parser("grpo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    print(f"Bắt đầu tác vụ {args.command}...", flush=True)
    if args.command in ("train", "infer"):
        from .runtime import infer, train
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if args.command == "train":
            train(cfg, args.data_dir, args.run_dir)
        else:
            infer(cfg, args.out, args.data_dir, args.split, args.checkpoint, args.batch_size, args.limit)
    elif args.command == "score":
        import hashlib
        from scripts.score import main as score_main
        command = ["--pred", args.pred, "--name", args.name, "--out", args.out,
                   "--splits-dir", str(Path(args.data_dir)/"splits"), "--split", args.split, "--coord", args.coord]
        command.extend(["--meta", "prediction_sha256="+hashlib.sha256(Path(args.pred).read_bytes()).hexdigest()])
        if args.checkpoint:
            manifest = json.loads((Path(args.checkpoint)/"manifest.json").read_text(encoding="utf-8"))
            meta = json.loads((Path(args.data_dir)/"splits"/"meta.json").read_text(encoding="utf-8"))
            if not manifest["completed"] or manifest["dataset_version"] != meta["dataset_version"]:
                raise ValueError("Invalid checkpoint for evidence")
            for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
                command.extend(["--meta", f"{key}={manifest[key]}"])
        if args.setup:
            command.extend(["--meta", f"setup={args.setup}"])
        score_main(command)
    elif args.command == "report":
        from .reporting import make_summary
        make_summary(args.evidence_dir, args.out, "REPORT.md" if args.fill_table else None)
    elif args.command == "figures":
        from .reporting import prediction_figures
        prediction_figures(args.pred, args.data_dir, args.out_dir, args.coord)
    else:
        from .grpo import train_grpo
        train_grpo(args.checkpoint, args.run_dir, args.data_dir)


if __name__ == "__main__":
    main()
