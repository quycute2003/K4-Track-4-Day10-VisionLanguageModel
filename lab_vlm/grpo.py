"""Optional E5 recipe for the pinned TRL 1.9.2; run after E0–E4."""
import gc
import json
import time
import uuid
from pathlib import Path

from .data import make_prompt, objects_to_answer, reward_iou, verify_splits, write_json
from .runtime import load_finetuned, require_gpu, save_checkpoint, seed_all


def grpo_reward(completions, gt_boxes, **kwargs):
    def text(completion):
        if isinstance(completion, str):
            return completion
        return "".join(message["content"] for message in completion)
    return [reward_iou(text(completion), gt) for completion, gt in zip(completions, gt_boxes)]


def train_grpo(checkpoint, run_dir, data_dir="data"):
    import importlib.metadata
    import yaml
    from datasets import Dataset, Image
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    torch = require_gpu()
    if importlib.metadata.version("trl") != "1.9.2":
        raise RuntimeError("E5 requires trl==1.9.2 as specified by the lab")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir/"run.json").exists():
        existing = json.loads((run_dir/"run.json").read_text(encoding="utf-8"))
        if existing.get("parent_checkpoint") != str(checkpoint):
            raise ValueError("E5 folder belongs to another SFT checkpoint; use a new folder")
        if existing.get("completed"):
            from .runtime import checkpoint_path
            return checkpoint_path(run_dir)
    options = yaml.safe_load(Path("configs/e5.yaml").read_text(encoding="utf-8"))["grpo"]
    splits, meta = verify_splits(data_dir)
    model, processor, parent = load_finetuned(checkpoint, trainable=True)
    if not parent["completed"] or parent["dataset_version"] != meta["dataset_version"]:
        raise ValueError("E5 requires a completed SFT checkpoint from this dataset")
    # Continue LoRA; keep the SFT projector fixed but save its learned weights with E5.
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
        if p.requires_grad:
            p.data = p.data.float()
    seed_all(parent["config"]["seed"])
    records = []
    for entry in splits["train"]:
        records.append({
            "prompt": [{"role": "user", "content": make_prompt(
                parent["config"].get("coord", "norm1000"), entry["width"], entry["height"])}],
            "image": str(Path(data_dir)/entry["image_path"]),
            "gt_boxes": json.loads(objects_to_answer(entry["objects"], parent["config"].get("coord", "norm1000"),
                                                     entry["width"], entry["height"])),
        })
    dataset = Dataset.from_list(records).cast_column("image", Image())
    cfg = dict(parent["config"], name="E5")
    manifest = dict(parent, config=cfg, completed=False, step=0, run_id=uuid.uuid4().hex,
                    parent_checkpoint=str(checkpoint), grpo=options, train_minutes=0, peak_vram_gb=0,
                    projector_trainable_params=0,
                    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write_json(run_dir/"config.json", cfg)
    parent_path = run_dir/"parent.json"
    parent_identity = {"checkpoint": str(checkpoint), "run_id": parent["run_id"], "grpo": options}
    if parent_path.exists() and json.loads(parent_path.read_text(encoding="utf-8")) != parent_identity:
        raise ValueError("E5 parent/config changed; use a new run directory")
    write_json(parent_path, parent_identity)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    class SaveProjector(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            cp = Path(args.output_dir)/f"checkpoint-{state.global_step}"
            from .runtime import is_projector
            if cfg.get("train_projector"):
                torch.save({n: p.detach().float().cpu() for n, p in model.named_parameters()
                            if is_projector(n)}, cp/"projector.pt")
            write_json(cp/"e5_parent.json", {"parent": str(checkpoint), "dataset_version": meta["dataset_version"]})
            elapsed = (time.perf_counter()-started)/60
            write_json(cp/"resource.json", {"train_minutes": minutes_before+elapsed,
                       "peak_vram_gb": max(peak_before, torch.cuda.max_memory_allocated()/1024**3)})

    args = GRPOConfig(
        output_dir=str(run_dir/"trainer"), num_generations=options["num_generations"],
        max_completion_length=options["max_completion_length"], max_steps=options["max_steps"],
        learning_rate=options["learning_rate"], beta=options["beta"], use_vllm=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True, bf16=False, report_to="none", remove_unused_columns=False,
        save_steps=20, logging_steps=1, seed=cfg["seed"], log_completions=True)
    trainer = GRPOTrainer(model=model, reward_funcs=grpo_reward, args=args, train_dataset=dataset,
                          processing_class=processor, callbacks=[SaveProjector()])
    trainer_checkpoints = sorted((run_dir/"trainer").glob("checkpoint-*"),
                                 key=lambda p: int(p.name.rsplit("-", 1)[1]))
    # TRL checkpoints hold optimizer/RNG/adapter; restore the separately persisted projector.
    latest = trainer_checkpoints[-1] if trainer_checkpoints else None
    resources = json.loads((latest/"resource.json").read_text()) if latest and (latest/"resource.json").exists() else {}
    minutes_before, peak_before = resources.get("train_minutes", 0), resources.get("peak_vram_gb", 0)
    if latest and cfg.get("train_projector"):
        from .runtime import restore_projector
        restore_projector(model, latest, True)
    trainer.train(resume_from_checkpoint=str(latest) if latest else None)
    torch.cuda.synchronize()
    manifest.update(step=trainer.state.global_step, completed=True,
                    train_minutes=minutes_before+(time.perf_counter()-started)/60,
                    peak_vram_gb=max(peak_before, torch.cuda.max_memory_allocated()/1024**3))
    write_json(run_dir/"grpo_log.json", trainer.state.log_history)
    final = save_checkpoint(model, processor, run_dir, manifest)
    del trainer, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return final
