"""Multimodal encoding and padding; imports Torch only when needed."""
from .data import make_prompt, objects_to_answer


def user_message(image, entry, coord):
    return {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": make_prompt(coord, entry["width"], entry["height"])},
    ]}


def encode_example(processor, image, entry, coord="norm1000"):
    import torch
    user = user_message(image, entry, coord)
    answer = objects_to_answer(entry["objects"], coord, entry["width"], entry["height"])
    kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt = processor.apply_chat_template([user], add_generation_prompt=True, **kwargs)
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": answer}], add_generation_prompt=False, **kwargs)
    n = prompt["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :n], prompt["input_ids"]):
        raise ValueError("Chat template mismatch: prompt is not a prefix of the full example")
    full.pop("token_type_ids", None)
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    if not torch.any(labels != -100):
        raise ValueError("No supervised answer tokens")
    full["labels"] = labels
    return dict(full)


def collate(batch, pad_token_id):
    import torch
    from torch.nn.functional import pad
    if not batch:
        raise ValueError("Empty batch")
    seq = {"input_ids": pad_token_id, "attention_mask": 0, "mm_token_type_ids": 0, "labels": -100}
    visual = {"pixel_values", "image_grid_thw"}
    keys = set(batch[0])
    if any(set(row) != keys for row in batch) or keys - set(seq) - visual:
        raise ValueError(f"Unexpected or inconsistent processor fields: {keys}")
    n = max(row["input_ids"].shape[1] for row in batch)
    result = {}
    for key in keys:
        values = [row[key] for row in batch]
        result[key] = (torch.cat([pad(v, (0, n-v.shape[1]), value=seq[key]) for v in values])
                       if key in seq else torch.cat(values, dim=0))
    return result
