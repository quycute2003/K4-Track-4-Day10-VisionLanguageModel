"""QLoRA SFT, resumable checkpoints and batched generation for Colab T4."""
import gc
import json
import math
import re
import time
import uuid
from pathlib import Path

from scripts.labcommon import read_jsonl
from .data import batch_indices, fingerprint, select_train, target_quality_report, verify_splits, write_json
from .encoding import collate, encode_example, user_message
from .progress import progress_stage

LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Chọn Runtime → Change runtime type → T4 GPU trên Colab.")
    return torch


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base(cfg):
    print("Kiểm tra GPU và chuẩn bị thư viện...", flush=True)
    torch = require_gpu()
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    print("GPU:", torch.cuda.get_device_name(), flush=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # Transformers 5.19 matches a prefix/exact/suffix, not arbitrary parent names.
        # Keep the full model.visual prefix as well as the legacy visual spelling.
        llm_int8_skip_modules=["visual", "model.visual", "lm_head"])
    with progress_stage(f"Tải/nạp model {cfg['model_id']} (NF4)..."):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], revision=cfg.get("revision"), quantization_config=quant,
            dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    with progress_stage("Nạp processor..."):
        processor = AutoProcessor.from_pretrained(
            cfg["model_id"], revision=getattr(model.config, "_commit_hash", None) or cfg.get("revision"))
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # Do not call prepare_model_for_kbit_training: it also upcasts frozen vision weights.
    quantized_vision = [n for n, m in model.named_modules()
                        if ("visual" in n or n == "lm_head") and "4bit" in type(m).__name__.lower()]
    if quantized_vision:
        raise RuntimeError(f"ViT/lm_head unexpectedly quantized: {quantized_vision[:3]}")
    return model, processor


def architecture_counts(model):
    counts = {"llm_and_head": 0, "vit_without_projector": 0, "projector": 0}
    for name, p in model.named_parameters():
        # Params4bit stores packed values; quant_state.shape gives original dimensions.
        shape = getattr(getattr(p, "quant_state", None), "shape", p.shape)
        key = "projector" if is_projector(name) else "vit_without_projector" if "visual." in name else "llm_and_head"
        counts[key] += math.prod(shape)
    counts["total"] = sum(counts.values())
    return counts


def setup_lora(model, cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    regex = LLM_TARGET if cfg["lora"]["targets"] == "llm" else f"(?:{LLM_TARGET}|{VIT_TARGET})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    expected = 252 if cfg["lora"]["targets"] == "llm" else 348
    if len(matched) != expected:
        raise RuntimeError(f"LoRA targets: got {len(matched)}, expected {expected}. Inspect named_modules().")
    counts = architecture_counts(model)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules=regex, task_type="CAUSAL_LM"))
    for name, p in model.named_parameters():
        if cfg.get("train_projector") and is_projector(name):
            p.requires_grad_(True)
        # GradScaler cannot unscale fp16 gradients; all trainable master weights are fp32.
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    diagnostics = {
        "architecture": counts, "matched_modules": matched,
        "n_matched_modules": len(matched),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "projector_trainable_params": sum(p.numel() for n, p in model.named_parameters()
                                           if p.requires_grad and is_projector(n)),
    }
    model.print_trainable_parameters()
    return model, diagnostics


def checkpoint_path(run_dir):
    run_dir = Path(run_dir)
    pointer = run_dir / "latest.json"
    if not pointer.exists():
        return None
    cp = run_dir / json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
    if not (cp / "manifest.json").exists():
        raise RuntimeError(f"Incomplete checkpoint: {cp}")
    return cp


def restore_projector(model, checkpoint, required):
    import torch
    path = Path(checkpoint) / "projector.pt"
    if not required:
        return
    if not path.exists():
        raise FileNotFoundError(f"Missing trained projector: {path}")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if is_projector(n)}
    if set(weights) != set(params):
        raise RuntimeError("Projector keys differ from checkpoint; refusing partial loading")
    with torch.no_grad():
        for name, p in params.items():
            p.data = p.data.float()
            p.copy_(weights[name].to(p.device))


def save_checkpoint(model, processor, run_dir, manifest, optimizer=None, scheduler=None, scaler=None):
    import torch
    run_dir = Path(run_dir)
    # A fresh folder avoids replacing a checkpoint that latest.json still references.
    cp = run_dir / f"checkpoint-{manifest['step']:04d}-{uuid.uuid4().hex[:8]}"
    cp.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(cp)
    processor.save_pretrained(cp)
    if manifest["config"].get("train_projector"):
        torch.save({n: p.detach().float().cpu() for n, p in model.named_parameters()
                    if is_projector(n)}, cp / "projector.pt")
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all()}, cp / "training_state.pt")
    write_json(cp / "manifest.json", manifest)
    write_json(run_dir / "latest.json", {"checkpoint": cp.name})
    write_json(run_dir / "run.json", manifest)
    return cp


def load_finetuned(checkpoint, trainable=False):
    from peft import PeftModel
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    from .environment import check_vit_lora_backend
    check_vit_lora_backend(manifest["config"])
    model, processor = load_base(manifest["config"])
    with progress_stage("Nạp adapter và projector từ checkpoint..."):
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=trainable)
        restore_projector(model, checkpoint, manifest["config"].get("train_projector"))
    return model, processor, manifest


def image_for(entry, data_dir):
    from PIL import Image
    with Image.open(Path(data_dir) / entry["image_path"]) as image:
        return image.convert("RGB")


def inspect_encoding(model, processor, entry, data_dir, coord="norm1000"):
    enc = encode_example(processor, image_for(entry, data_dir), entry, coord)
    image_tokens = int((enc["input_ids"] == model.config.image_token_id).sum())
    grid_tokens = int(enc["image_grid_thw"].prod(dim=1).sum()) // model.config.vision_config.spatial_merge_size**2
    if image_tokens != grid_tokens:
        raise ValueError("Image token count does not match the processed image grid")
    if (entry["width"], entry["height"]) == (640, 360) and image_tokens != 220:
        raise ValueError(f"Expected 220 vision tokens; got {image_tokens}. Check processor resizing.")
    decoded = processor.tokenizer.decode(enc["labels"][enc["labels"] != -100].tolist())
    print("Vision tokens:", image_tokens, "\nSupervised answer:", decoded)
    return {"image_tokens": image_tokens, "supervised_answer": decoded}


def train(cfg, data_dir="data", run_dir="work/e1", resume=True):
    if any(cfg[k] <= 0 for k in ("max_steps", "batch_size", "grad_accum", "lr")) or cfg.get("checkpoint_every", 100) <= 0:
        raise ValueError("Training steps, batch, accumulation, lr and checkpoint interval must be positive")
    from .environment import check_vit_lora_backend
    check_vit_lora_backend(cfg)
    torch = require_gpu()
    from transformers import get_cosine_schedule_with_warmup
    splits, meta = verify_splits(data_dir)
    rows = select_train(splits["train"], cfg.get("train_fraction", 1), cfg["seed"])
    quality = target_quality_report(rows, cfg.get("coord", "norm1000"))
    print(f"Nhãn train: {quality['target_objects']}/{quality['source_objects']} box có diện tích; "
          f"bỏ {quality['zero_area_annotations_skipped']} nhãn diện tích 0.", flush=True)
    seed_all(cfg["seed"])
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "target_quality.json", quality)
    run_hash = fingerprint({"config": cfg, "dataset_version": meta["dataset_version"],
                            "train_ids": [r["image_id"] for r in rows]})
    identity_path = run_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8"))["run_hash"] != run_hash:
        raise ValueError("This run folder already uses another configuration; use a new directory")
    write_json(identity_path, {"run_hash": run_hash})
    previous = checkpoint_path(run_dir)
    if previous:
        old = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
        if old["run_hash"] != run_hash:
            raise ValueError("Config/data changed. Use a new run directory to preserve the old run.")
        if not resume:
            raise ValueError("Run exists; use resume or a new directory")
        if old.get("completed"):
            print("Already completed:", previous)
            return previous
    else:
        old = {}
    model, processor = load_base(cfg)
    encoding = inspect_encoding(model, processor, rows[0], data_dir, cfg.get("coord", "norm1000"))
    model, diagnostics = setup_lora(model, cfg)
    if previous:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        set_peft_model_state_dict(model, load_peft_weights(previous, device="cpu"))
        restore_projector(model, previous, cfg.get("train_projector"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, math.ceil(0.05*cfg["max_steps"])), num_training_steps=cfg["max_steps"])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    start_step = old.get("step", 0)
    if previous:
        state = torch.load(previous / "training_state.pt", map_location="cpu", weights_only=True)
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        del state
    manifest = {
        **diagnostics, "encoding_check": encoding, "config": dict(cfg), "run_hash": run_hash,
        "target_quality": quality,
        "run_id": old.get("run_id", uuid.uuid4().hex), "dataset_version": meta["dataset_version"],
        "train_images": len(rows), "step": start_step, "completed": False,
        "train_minutes": old.get("train_minutes", 0), "peak_vram_gb": old.get("peak_vram_gb", 0),
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
    }
    write_json(run_dir / "config.json", cfg)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    minutes_before = manifest["train_minutes"]
    log_path = run_dir / "loss.jsonl"
    accum = cfg["grad_accum"]
    model.train()
    opt.zero_grad(set_to_none=True)
    step = start_step
    overflow_retries = 0
    while step < cfg["max_steps"]:
        loss_total = 0.0
        for micro in range(accum):
            indices = batch_indices(len(rows), cfg["batch_size"], step*accum+micro, cfg["seed"])
            examples = [encode_example(processor, image_for(rows[i], data_dir), rows[i],
                                       cfg.get("coord", "norm1000")) for i in indices]
            batch = {k: v.to("cuda") for k, v in collate(examples, processor.tokenizer.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step+1}; restart with smaller lr in a new run.")
            loss_total += float(loss.detach()) / accum
            scaler.scale(loss / accum).backward()
            del batch, examples, loss
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm):
            # GradScaler skips the overflowing update and lowers its scale. Do not count it
            # as an optimizer step or advance the scheduler/deterministic sample schedule.
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            overflow_retries += 1
            if overflow_retries >= 8:
                raise FloatingPointError(f"Persistent gradient overflow at step {step+1}; reduce lr in a new run.")
            print(f"AMP overflow: retry step {step+1}, scale={scaler.get_scale()}", flush=True)
            continue
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        manifest["step"] = step+1
        record = {"step": step+1, "loss": loss_total, "lr": scheduler.get_last_lr()[0]}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
        if (step+1) % 10 == 0 or step == start_step:
            print(f"{cfg['name']} step {step+1}/{cfg['max_steps']} loss={loss_total:.4f}", flush=True)
        if (step+1) % cfg.get("checkpoint_every", 100) == 0 or step+1 == cfg["max_steps"]:
            torch.cuda.synchronize()
            manifest["train_minutes"] = minutes_before + (time.perf_counter()-started)/60
            manifest["peak_vram_gb"] = max(manifest["peak_vram_gb"], torch.cuda.max_memory_allocated()/1024**3)
            manifest["completed"] = step+1 == cfg["max_steps"]
            previous = save_checkpoint(model, processor, run_dir, manifest, opt, scheduler, scaler)
            print("Saved:", previous, flush=True)
        step += 1
        overflow_retries = 0
    del model, processor, opt, scheduler, scaler, params
    gc.collect()
    torch.cuda.empty_cache()
    return previous


def infer(cfg, out, data_dir="data", split="test", checkpoint=None, batch_size=4, limit=None):
    print(f"{cfg['name']}: chuẩn bị suy luận trên {split}...", flush=True)
    torch = require_gpu()
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    if limit is not None:
        if split == "test":
            raise ValueError("Use val for a limited debug run; test must include every image")
        rows = rows[:limit]
    dataset_meta = json.loads((Path(data_dir) / "splits" / "meta.json").read_text(encoding="utf-8"))
    if checkpoint:
        saved = json.loads((Path(checkpoint) / "manifest.json").read_text(encoding="utf-8"))
        if saved["dataset_version"] != dataset_meta["dataset_version"] or not saved["completed"]:
            raise ValueError("Checkpoint incomplete or uses a different dataset")
        cfg = saved["config"]
        model_key = {"run_id": saved["run_id"], "step": saved["step"]}
    else:
        model_key = cfg
    identity = {"model": model_key, "split": split, "dataset_version": dataset_meta["dataset_version"],
                "image_ids": [r["image_id"] for r in rows], "coord": cfg.get("coord", "norm1000"),
                "max_new_tokens": cfg.get("max_new_tokens", 256)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".meta.json")
    if sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint(identity):
            raise ValueError(f"Prediction file belongs to another run: {out}")
    elif out.exists() and out.stat().st_size:
        raise ValueError("Existing predictions have no provenance; use a different output filename")
    else:
        write_json(sidecar, {"fingerprint": fingerprint(identity), **identity})
    done = {}
    if out.exists():
        # Recover only a truncated last line caused by interruption; other damage is an error.
        with out.open("rb+") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(position)
                    break
                row = json.loads(line)
                if row["image_id"] in done:
                    raise ValueError("Duplicate prediction IDs")
                done[row["image_id"]] = row["raw"]
    if set(done) - set(identity["image_ids"]):
        raise ValueError("Prediction IDs are outside this split")
    pending = [r for r in rows if r["image_id"] not in done]
    print(f"{cfg['name']} {split}: {len(done)}/{len(rows)} ảnh đã lưu; {len(pending)} ảnh còn lại.", flush=True)
    if not pending:
        print("Predictions already complete:", out)
        return out
    if checkpoint:
        model, processor, _ = load_finetuned(checkpoint)
    else:
        model, processor = load_base(cfg)
    model.eval()
    model.config.use_cache = True
    position = 0
    started = time.monotonic()
    while position < len(pending):
        chunk = pending[position:position+batch_size]
        try:
            messages = [[user_message(image_for(r, data_dir), r, cfg.get("coord", "norm1000"))] for r in chunk]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", processor_kwargs={"padding": True})
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            label = f"{cfg['name']} {split}: đang sinh kết quả cho ảnh {len(done)+position+1}–{len(done)+position+len(chunk)}/{len(rows)}"
            with progress_stage(label), torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output = model.generate(**inputs, max_new_tokens=cfg.get("max_new_tokens", 256), do_sample=False)
            raw = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
            del inputs, output
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            inputs = output = None
            gc.collect()
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size//2)
            print("Inference batch reduced to", batch_size, flush=True)
            continue
        with out.open("a", encoding="utf-8") as stream:
            for entry, text in zip(chunk, raw):
                stream.write(json.dumps({"image_id": entry["image_id"], "raw": text}, ensure_ascii=False)+"\n")
            stream.flush()
        position += len(chunk)
        elapsed = time.monotonic()-started
        remaining = (len(pending)-position)*elapsed/position
        print(f"{cfg['name']} {split}: {len(done)+position}/{len(rows)} đã lưu; "
              f"còn khoảng {remaining/60:.1f} phút (ước tính từ các batch đã chạy).", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return out
