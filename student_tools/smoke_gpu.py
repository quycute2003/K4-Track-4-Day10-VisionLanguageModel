"""Two-step QLoRA E3 smoke test on a SMALL RANDOM model, not lab evidence.

Requires the pinned dependencies and CUDA. Everything is written under ignored work/.
Uses real Qwen3-VL modules, processor, NF4 and the same SFT/checkpoint/inference code.
"""
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    from lab_vlm.data import write_json
    from lab_vlm.runtime import infer, train
    from scripts.labcommon import write_jsonl
    root = ROOT/"work"/f"synthetic-smoke-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True)
    source = "Qwen/Qwen3-VL-4B-Instruct"
    cfg = AutoConfig.from_pretrained(source)
    cfg.text_config.hidden_size = 64
    cfg.text_config.intermediate_size = 128
    cfg.text_config.num_attention_heads = 4
    cfg.text_config.num_key_value_heads = 2
    cfg.text_config.head_dim = 16
    cfg.text_config.rope_parameters["mrope_section"] = [2, 3, 3]
    cfg.vision_config.hidden_size = 32
    cfg.vision_config.intermediate_size = 64
    cfg.vision_config.num_heads = 4
    cfg.vision_config.out_hidden_size = 64
    torch.manual_seed(42)
    tiny = Qwen3VLForConditionalGeneration(cfg)
    tiny.save_pretrained(root/"base")
    processor = AutoProcessor.from_pretrained(source)
    processor.save_pretrained(root/"base")
    del tiny, processor
    data = root/"data"
    (data/"images").mkdir(parents=True)
    Image.new("RGB", (640, 360), "gray").save(data/"images"/"blank.jpg")
    counts = {}
    for split in ("train", "val", "test"):
        write_jsonl(data/"splits"/f"{split}.jsonl", [{
            "image_id": f"synthetic-{split}", "image_path": "images/blank.jpg", "session": split,
            "width": 640, "height": 360,
            "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}],
        }])
        counts[split] = {"images": 1, "boxes": {"pothole": 1, "crack": 0, "manhole": 0}}
    write_json(data/"splits/meta.json", {"dataset_version": "synthetic-smoke-only", "counts": counts})
    run_config = {"name": "SYNTHETIC_SMOKE_ONLY", "model_id": str(root/"base"),
                  "lora": {"r": 4, "alpha": 8, "dropout": 0.05, "targets": "llm+vit"},
                  "train_projector": True, "lr": 2e-4, "batch_size": 1, "grad_accum": 1,
                  "max_steps": 2, "checkpoint_every": 1, "seed": 42, "coord": "norm1000", "max_new_tokens": 4}
    import lab_vlm.runtime as runtime
    original_save = runtime.save_checkpoint
    def interrupt_after_first_save(*args, **kwargs):
        checkpoint = original_save(*args, **kwargs)
        if args[3]["step"] == 1:
            raise InterruptedError("Simulated disconnect after a complete checkpoint")
        return checkpoint
    runtime.save_checkpoint = interrupt_after_first_save
    try:
        train(run_config, data, root/"run")
    except InterruptedError:
        print("Resuming simulated interrupted training.")
    finally:
        runtime.save_checkpoint = original_save
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = train(run_config, data, root/"run")
    manifest = json.loads((checkpoint/"manifest.json").read_text())
    assert manifest["completed"] and manifest["step"] == 2 and manifest["n_matched_modules"] == 348
    assert (checkpoint/"projector.pt").exists() and (checkpoint/"training_state.pt").exists()
    assert train(run_config, data, root/"run") == checkpoint
    uninterrupted = train(run_config, data, root/"uninterrupted")
    from safetensors.torch import load_file
    resumed_adapters = load_file(checkpoint/"adapter_model.safetensors")
    uninterrupted_adapters = load_file(uninterrupted/"adapter_model.safetensors")
    assert set(resumed_adapters) == set(uninterrupted_adapters)
    assert all(torch.allclose(value, uninterrupted_adapters[key], atol=1e-6, rtol=1e-4)
               for key, value in resumed_adapters.items()), "Resumed adapters diverge from uninterrupted training"
    resumed_projector = torch.load(checkpoint/"projector.pt", weights_only=True)
    uninterrupted_projector = torch.load(uninterrupted/"projector.pt", weights_only=True)
    assert all(torch.allclose(value, uninterrupted_projector[key], atol=1e-6, rtol=1e-4)
               for key, value in resumed_projector.items()), "Resumed projector differs"
    first_checkpoint = sorted((root/"run").glob("checkpoint-0001-*"))[0]
    first_adapters = load_file(first_checkpoint/"adapter_model.safetensors")
    assert any(not torch.equal(value, first_adapters[key]) for key, value in resumed_adapters.items())
    print("Resume matches uninterrupted training; adapter weights actually changed.")
    # Test generation on two images to exercise batched processor padding.
    from scripts.labcommon import read_jsonl
    entries = read_jsonl(data/"splits/test.jsonl")
    entries.append(dict(entries[0], image_id="synthetic-test-2"))
    write_jsonl(data/"splits/test.jsonl", entries)
    output = root/"predictions.jsonl"
    infer(run_config, output, data, checkpoint=checkpoint, batch_size=2)
    before = output.read_bytes()
    infer(run_config, output, data, checkpoint=checkpoint, batch_size=2)
    assert output.read_bytes() == before
    print("Small random-model GPU train/save/reload/generation smoke passed.")
    print("Synthetic checks only; no road-damage metrics/evidence were produced.")


if __name__ == "__main__":
    main()
