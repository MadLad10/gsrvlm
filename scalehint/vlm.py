"""Frozen VLM wrapper (Qwen3-VL by default; any HF image-text-to-text chat model).

Frames go in as a list of images rather than a video. This is the same 16
frames for every condition and avoids video-processor quirks (timestamp
metadata, temporal patch merging) that differ between transformers versions.
"""

import os

import torch
from PIL import Image


def pick_dtype(name):
    if name != "auto":
        return {"float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32}[name]
    major, _ = torch.cuda.get_device_capability()
    # Ampere+ has native bf16. Turing (T4) does not: try fp16 first.
    return torch.bfloat16 if major >= 8 else torch.float16


def load(model_id, dtype, device="cuda"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
    return model, proc


def load_frames(paths, side):
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = side / max(w, h)
        if s < 1:
            im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
        out.append(im)
    return out


def _inputs(model, proc, frames, prompt):
    content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return proc(text=[text], images=frames, return_tensors="pt").to(model.device)


@torch.no_grad()
def finite_check(model, proc, frames, prompt):
    """True if a forward pass gives finite logits (fp16 on T4 can overflow)."""
    inputs = _inputs(model, proc, frames, prompt)
    logits = model(**inputs).logits
    return bool(torch.isfinite(logits[:, -1]).all())


@torch.no_grad()
def answer(model, proc, frames, prompt, max_new_tokens=32):
    inputs = _inputs(model, proc, frames, prompt)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None)
    new = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(new, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------- vision tokens

def _per_image(out, n, counts):
    """Normalise whatever get_image_features returned into n tensors [tokens, D]."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        out = out.pooler_output
    elif hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    if isinstance(out, (list, tuple)):
        if len(out) == n and all(torch.is_tensor(t) for t in out):
            return list(out)
        # e.g. (image_embeds, deepstack_embeds): take the first usable element
        for item in out:
            try:
                return _per_image(item, n, counts)
            except ValueError:
                continue
        raise ValueError("no per-image features found")
    if torch.is_tensor(out):
        if out.dim() == 3 and out.shape[0] == n:
            return list(out)
        if out.dim() == 2 and out.shape[0] == sum(counts):
            return list(torch.split(out, counts))
    raise ValueError(f"unexpected feature shape {getattr(out, 'shape', type(out))}")


@torch.no_grad()
def vision_features(model, proc, frames):
    """Mean-pooled vision-encoder output (post-merger, i.e. what the LLM sees),
    one vector per frame -> [n_frames, D] float16 on CPU."""
    ip = proc.image_processor(images=frames, return_tensors="pt")
    pv = ip["pixel_values"].to(model.device, model.dtype)
    thw = ip["image_grid_thw"].to(model.device)
    fn = getattr(model, "get_image_features", None)
    if fn is None:
        fn = model.model.get_image_features
    out = fn(pv, thw)
    merge = getattr(getattr(model.config, "vision_config", None), "spatial_merge_size", 2)
    counts = [int(c) for c in (thw.prod(-1) // (merge ** 2)).tolist()]
    feats = _per_image(out, len(frames), counts)
    return torch.stack([f.float().mean(0) for f in feats]).half().cpu()


def env_info(model_id, dtype):
    import transformers
    info = {"model": model_id, "dtype": str(dtype).replace("torch.", ""),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
