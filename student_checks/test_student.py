"""Checks for student code, separate from the instructor-owned tests/."""
import json
import re
from copy import deepcopy

import pytest

from lab_vlm.data import batch_indices, make_prompt, objects_to_answer, reward_iou, select_train, yolo_to_box
from lab_vlm.runtime import LLM_TARGET, VIT_TARGET, is_projector


def test_coordinates_and_roundtrip():
    box = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4])
    assert box["bbox_2d"] == [400, 300, 600, 700]
    pixel = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4], "pixel")
    assert pixel["bbox_2d"] == [256, 108, 384, 252]
    assert [pixel["bbox_2d"][i]/(640 if i%2 == 0 else 360)*1000 for i in range(4)] == box["bbox_2d"]
    assert yolo_to_box("crack", [0, 0, 0.2, 0.2])["bbox_2d"] == [0, 0, 100, 100]


def test_zero_area_source_annotation_is_omitted_without_mutating_ground_truth():
    objects = [{"label": "crack", "yolo": [0.639062, 0.786111, 0.003125, 0.0]},
               {"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]
    original = deepcopy(objects)
    answer = json.loads(objects_to_answer(objects))
    assert len(answer) == 1 and answer[0]["label"] == "pothole"
    assert objects == original
    assert objects_to_answer(objects[:1]) == "[]"
    with pytest.raises(ValueError, match="Zero-area"):
        yolo_to_box("crack", objects[0]["yolo"])


def test_positive_subunit_boxes_keep_area_at_boundaries_in_both_scales():
    for coord in ("norm1000", "pixel"):
        sx, sy = (1000, 1000) if coord == "norm1000" else (640, 360)
        for yolo in ([0.639, 0.786, 0.004, 0.0004], [0, 0, 0.0001, 0.0001],
                     [1, 1, 0.0001, 0.0001]):
            bb = yolo_to_box("crack", yolo, coord)["bbox_2d"]
            assert 0 <= bb[0] < bb[2] <= sx and 0 <= bb[1] < bb[3] <= sy
            assert all(isinstance(v, int) for v in bb)
    assert yolo_to_box("crack", [0.639, 0.786, 0.004, 0.0004])["bbox_2d"] == [637, 785, 641, 787]


def test_target_quality_counts_invalid_and_rounding_repairs():
    from lab_vlm.data import target_quality_report
    entries = [{"image_id": "bad-annotation", "width": 640, "height": 360,
                "objects": [{"label": "crack", "yolo": [0.639062, 0.786111, 0.003125, 0]},
                            {"label": "crack", "yolo": [0.639, 0.786, 0.004, 0.0004]}]}]
    report = target_quality_report(entries)
    assert report["source_objects"] == 2 and report["target_objects"] == 1
    assert report["zero_area_annotations_skipped"] == 1
    assert report["positive_boxes_repaired_after_rounding"] == 1
    assert report["zero_area_annotations"][0]["image_id"] == "bad-annotation"


def test_sorted_answers_and_empty():
    objects = [{"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
               {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    assert json.loads(objects_to_answer(objects))[0]["label"] == "pothole"
    assert objects_to_answer([]) == "[]"
    assert "0–1000" in make_prompt() and "640×360" in make_prompt("pixel")


def test_target_regex_excludes_mergers_embeddings_and_other_layers():
    assert re.fullmatch(LLM_TARGET, "model.language_model.layers.35.self_attn.q_proj")
    assert re.fullmatch(VIT_TARGET, "model.visual.blocks.23.mlp.linear_fc2")
    assert not re.fullmatch(VIT_TARGET, "model.visual.merger.linear_fc2")
    assert not re.fullmatch(LLM_TARGET, "model.language_model.embed_tokens")
    assert is_projector("base_model.model.model.visual.deepstack_merger_list.2.linear_fc2.weight")
    assert not is_projector("base_model.model.model.visual.blocks.0.mlp.linear_fc2.weight")


def test_subset_and_resume_batch_schedule():
    rows = list(range(101))
    small, full = select_train(rows, 0.1), select_train(rows)
    assert len(small) == 10 and small == full[:10]
    assert rows == list(range(101))
    batches = [batch_indices(7, 2, step) for step in range(8)]
    assert sorted(sum(batches[:4], [])) == list(range(7))
    assert batches[4:] == [batch_indices(7, 2, step) for step in range(4, 8)]
    assert batches[:4] != batches[4:]


def test_reward_perfect_invalid_wrong_class_extra_and_empty():
    gt = [{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}]
    assert reward_iou(json.dumps(gt), gt) == 1
    assert reward_iou("bad json", gt) == 0
    wrong = deepcopy(gt)
    wrong[0]["label"] = "crack"
    assert reward_iou(json.dumps(wrong), gt) == pytest.approx(0.2)
    assert reward_iou(json.dumps(gt+gt), gt) == pytest.approx(0.6)
    assert reward_iou("[]", []) == 1
    assert reward_iou("[]", gt) == pytest.approx(0.2)
    assert reward_iou(json.dumps(gt), []) == pytest.approx(0.2)
    giant = [{"label": "pothole", "bbox_2d": [0, 0, 1000, 1000]}]
    assert reward_iou(json.dumps(giant), gt) == pytest.approx(0.2)


def test_grpo_reward_accepts_text_and_conversational_completions():
    from lab_vlm.grpo import grpo_reward
    gt = [{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}]
    text = json.dumps(gt)
    assert grpo_reward([text, [{"role": "assistant", "content": text}]], [gt, gt]) == [1, 1]


def test_collate_masks_padding_and_concatenates_vision():
    torch = pytest.importorskip("torch")
    from lab_vlm.encoding import collate
    first = {"input_ids": torch.tensor([[10, 11, 12]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
             "labels": torch.tensor([[-100, -100, 12]]), "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
             "pixel_values": torch.ones(4, 6), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    second = {"input_ids": torch.tensor([[20, 21]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
              "labels": torch.tensor([[-100, 21]]), "mm_token_type_ids": torch.tensor([[1, 0]]),
              "pixel_values": torch.ones(8, 6), "image_grid_thw": torch.tensor([[1, 2, 4]])}
    result = collate([first, second], 0)
    assert result["input_ids"].tolist() == [[10, 11, 12], [20, 21, 0]]
    assert result["labels"].tolist() == [[-100, -100, 12], [-100, 21, -100]]
    assert result["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert result["pixel_values"].shape == (12, 6)
    assert result["image_grid_thw"].shape == (2, 3)


def test_encoding_masks_only_prompt_and_checks_prefix():
    torch = pytest.importorskip("torch")
    from lab_vlm.encoding import encode_example
    class Processor:
        mismatch = False
        def apply_chat_template(self, messages, **kwargs):
            tokens = [1, 2, 3] if len(messages) == 1 else [1, 2, 3, 8, 9]
            if self.mismatch and len(messages) > 1:
                tokens[0] = 99
            return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones(1, len(tokens)),
                    "token_type_ids": torch.zeros(1, len(tokens))}
    p = Processor()
    entry = {"width": 640, "height": 360, "objects": []}
    encoded = encode_example(p, None, entry)
    assert encoded["labels"].tolist() == [[-100, -100, -100, 8, 9]]
    assert "token_type_ids" not in encoded
    p.mismatch = True
    with pytest.raises(ValueError, match="prefix"):
        encode_example(p, None, entry)


def test_checkpoint_projector_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from lab_vlm.runtime import restore_projector
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Module()
            self.visual.merger = torch.nn.Linear(2, 2).half()
    model = Model()
    before = {n: p.detach().float().clone() for n, p in model.named_parameters()}
    torch.save(before, tmp_path/"projector.pt")
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    restore_projector(model, tmp_path, True)
    for n, p in model.named_parameters():
        assert p.dtype == torch.float32 and torch.equal(p, before[n])
    torch.save({}, tmp_path/"projector.pt")
    with pytest.raises(RuntimeError, match="keys"):
        restore_projector(model, tmp_path, True)


def test_score_report_figures_and_clean_package(tmp_path):
    from PIL import Image
    from lab_vlm.cli import main
    from lab_vlm.data import write_json
    from lab_vlm.packaging import package_project
    from lab_vlm.reporting import make_summary, prediction_figures
    from scripts.labcommon import write_jsonl
    data = tmp_path/"data"
    (data/"images").mkdir(parents=True)
    rows = []
    for i in range(6):
        Image.new("RGB", (640, 360), "gray").save(data/"images"/f"{i}.jpg")
        rows.append({"image_id": str(i), "image_path": f"images/{i}.jpg", "width": 640, "height": 360,
                     "session": "synthetic", "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.4]}]})
    write_jsonl(data/"splits/test.jsonl", rows)
    write_json(data/"splits/meta.json", {"dataset_version": "synthetic-test-only"})
    pred = tmp_path/"work"/"pred.jsonl"
    write_jsonl(pred, [{"image_id": r["image_id"], "raw": objects_to_answer(r["objects"])} for r in rows])
    evidence = tmp_path/"report/evidence/e0.json"
    main(["score", "--pred", str(pred), "--data-dir", str(data), "--out", str(evidence), "--name", "E0"])
    scored = json.loads(evidence.read_text(encoding="utf-8"))
    assert scored["micro"]["f1"] == 1 and scored["missing_predictions"] == 0
    assert len(scored["meta"]["prediction_sha256"]) == 64
    report = tmp_path/"REPORT.md"
    report.write_text("`dataset_version`: <điền>\n\n| Mã | Setup |\n|---|---|\n| E0 | |\n\n### Câu 1\n<điền>\n", encoding="utf-8")
    make_summary(evidence.parent, tmp_path/"report/SUMMARY.md", report)
    assert "1.0000" in report.read_text(encoding="utf-8") and "<điền>" in report.read_text(encoding="utf-8")
    figures = prediction_figures(pred, data, tmp_path/"report/figures")
    assert len(list(figures.glob("prediction_*.png"))) == 6
    assert Image.open(figures/"prediction_01.png").size == (1280, 430)
    (tmp_path/"lab_vlm").mkdir()
    (tmp_path/"lab_vlm/secret.pt").write_bytes(b"excluded")
    (tmp_path/"lab_vlm/valid.py").write_text("x = 1")
    bundle = package_project(tmp_path, tmp_path/"output.zip", include_report=True)
    import zipfile
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert "lab_vlm/valid.py" in names and "lab_vlm/secret.pt" not in names
        assert not any(name.startswith(("data/", "work/")) for name in names)


def test_notebook_runner_displays_stdout_stderr_and_reports_failure(tmp_path, capsys):
    import subprocess
    from lab_vlm.execution import run_module
    (tmp_path/"stream_fixture.py").write_text(
        "import sys\nprint('stdout message', flush=True)\nprint('stderr message', file=sys.stderr, flush=True)\nsys.exit(3)\n")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_module("stream_fixture", cwd=tmp_path)
    assert caught.value.returncode == 3
    displayed = capsys.readouterr().out
    assert "stdout message" in displayed and "stderr message" in displayed


def test_notebook_runner_stops_child_when_cell_is_interrupted(monkeypatch):
    import lab_vlm.execution as execution
    class InterruptedStream:
        closed = False
        def __iter__(self):
            raise KeyboardInterrupt
        def close(self):
            self.closed = True
    class Process:
        stdout = InterruptedStream()
        stopped = False
        def poll(self):
            return None if not self.stopped else -9
        def kill(self):
            self.stopped = True
        def wait(self):
            return -9
    child = Process()
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(KeyboardInterrupt):
        execution.run_module("unused")
    assert child.stopped and child.stdout.closed


def test_vit_lora_checks_optional_torchao_before_loading_model(monkeypatch):
    import lab_vlm.environment as environment
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: "0.10.0")
    environment.check_vit_lora_backend({"lora": {"targets": "llm"}})
    with pytest.raises(RuntimeError, match="torchao 0.10.0"):
        environment.check_vit_lora_backend({"lora": {"targets": "llm+vit"}})
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: "0.16.0")
    environment.check_vit_lora_backend({"lora": {"targets": "llm+vit"}})
    def missing(name):
        raise environment.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(environment.importlib.metadata, "version", missing)
    environment.check_vit_lora_backend({"lora": {"targets": "llm+vit"}})


def test_colab_torchao_cleanup_removes_only_incompatible_optional_package(monkeypatch):
    import lab_vlm.environment as environment
    import subprocess
    calls = []
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: "0.10.0")
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0, stdout="Uninstalled old torchao")
    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment.remove_incompatible_colab_torchao() is True
    assert calls[0][-3:] == ["uninstall", "-y", "torchao"]
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: "0.16.0")
    assert environment.remove_incompatible_colab_torchao() is False
    assert len(calls) == 1


def test_cpu_export_reuses_saved_artifacts_and_preserves_evidence(tmp_path, monkeypatch):
    import hashlib
    import zipfile
    from lab_vlm.data import write_json
    from lab_vlm.export_saved import finish_saved_results
    from lab_vlm.packaging import package_lightning_project
    from scripts.labcommon import write_jsonl
    from scripts.score import evaluate
    from PIL import Image
    source, root = tmp_path/"saved", tmp_path/"repo"
    root.mkdir()
    (root/"REPORT.md").write_text("`dataset_version`: <điền>\n\n| Mã | Setup |\n|---|---|\n| E0 | |\n\n### Câu 1\n<điền>\n", encoding="utf-8")
    entry = {"image_id": "synthetic", "width": 640, "height": 360,
             "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]}
    for slug in ("e0", "e1", "e2", "e3"):
        pred = source/"work"/f"{slug}_test.jsonl"
        raw = objects_to_answer(entry["objects"])
        write_jsonl(pred, [{"image_id": "synthetic", "raw": raw}])
        scored = evaluate([entry], {"synthetic": raw}, n_boot=40)
        scored.update(score_version=1, split="test", coord="norm1000", name=slug.upper(),
                      dataset_version="synthetic-only", meta={"prediction_sha256": hashlib.sha256(pred.read_bytes()).hexdigest()})
        write_json(source/"report/evidence"/f"{slug}.json", scored)
        if slug != "e0":
            write_json(source/"report/validation"/f"{slug}.json",
                       {"split": "val", "dataset_version": "synthetic-only", "micro": {"f1": {"e1": 0.2, "e2": 0.4, "e3": 0.3}[slug]}})
    figures = source/"report/figures"
    figures.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (10, 10)).save(figures/f"prediction_{i+1:02d}.png")
    write_json(source/"report/diagnostics/e0_wrong_coord.json", {"micro": {"f1": 0}})
    # A cached artifact export must not try to download the dataset or load models.
    monkeypatch.setattr("urllib.request.urlretrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected download")))
    output = finish_saved_results(root, source, fetch_data=False)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "report/evidence/e3.json" in names
        assert not any(n.startswith(("work/", "data/")) for n in names)
        assert archive.read("report/evidence/e1.json") == (source/"report/evidence/e1.json").read_bytes()
    state = json.loads((root/"report/diagnostics/export_state.json").read_text())
    assert state["figure_model"] == "e2" and state["training_or_inference_performed"] is False
    assert state["e4_evidence_present"] is False
    migration = package_lightning_project(root, source, tmp_path/"lightning-ready.zip")
    with zipfile.ZipFile(migration) as archive:
        assert "work/e0_test.jsonl" in archive.namelist()
        assert not any(n.endswith((".pt", ".bin", ".safetensors")) for n in archive.namelist())


def test_cpu_export_refuses_mismatched_saved_predictions(tmp_path):
    from lab_vlm.data import write_json
    from lab_vlm.export_saved import finish_saved_results
    from scripts.labcommon import write_jsonl
    root = tmp_path
    (root/"REPORT.md").write_text("Report")
    for slug in ("e0", "e1", "e2", "e3"):
        write_json(root/"report/evidence"/f"{slug}.json", {"score_version": 1, "split": "test",
            "coord": "norm1000", "missing_predictions": 0, "n_images": 1, "dataset_version": "test",
            "meta": {"prediction_sha256": "wrong"}})
    write_jsonl(root/"work/e1_test.jsonl", [{"image_id": "x", "raw": "[]"}])
    with pytest.raises(ValueError, match="hash"):
        finish_saved_results(root, fetch_data=False)
