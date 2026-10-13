"""Verify real HF configuration, PEFT targets and processor without downloading model weights.

Uses a meta-device model and a synthetic blank image; produces no experiment evidence.
Run with the pinned Colab dependencies installed: py student_tools/verify_compatibility.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    from lab_vlm.encoding import collate, encode_example
    from lab_vlm.runtime import LLM_TARGET, VIT_TARGET, architecture_counts
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    cfg = AutoConfig.from_pretrained(model_id)
    revision = getattr(cfg, "_commit_hash", None)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(cfg)
    names = [n for n, _ in model.named_modules()]
    llm = [n for n in names if re.fullmatch(LLM_TARGET, n)]
    vit = [n for n in names if re.fullmatch(VIT_TARGET, n)]
    assert len(llm) == 252 and len(vit) == 96, (len(llm), len(vit))
    print("Architecture:", json.dumps(architecture_counts(model), indent=2))
    with torch.device("meta"):
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
            target_modules=f"(?:{LLM_TARGET}|{VIT_TARGET})", task_type="CAUSAL_LM"))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count > 0
    print("E3 LoRA parameters:", count)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    entry = {"width": 640, "height": 360, "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]}
    encoded = encode_example(processor, Image.new("RGB", (640, 360)), entry)
    image_tokens = int((encoded["input_ids"] == cfg.image_token_id).sum())
    assert image_tokens == 220, image_tokens
    supervised = processor.tokenizer.decode(encoded["labels"][encoded["labels"] != -100].tolist())
    assert supervised.strip().endswith("<|im_end|>")
    assert json.loads(supervised.strip().removesuffix("<|im_end|>").strip())[0]["label"] == "pothole"
    batch = collate([encoded, encoded], processor.tokenizer.pad_token_id)
    assert batch["image_grid_thw"].shape == (2, 3)
    print("Processor keys:", sorted(encoded))
    print("Vision tokens:", image_tokens)
    print("Supervised tokens:", supervised)
    print("Pinned API/processor/LoRA compatibility verified; no model training performed.")


if __name__ == "__main__":
    main()
