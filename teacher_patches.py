"""MapAnything in 16-bit on Turing GPUs (T4 = sm_75, same as the RTX 2060).

Ported from ThriftSplat's lowvram.py, where each of these was measured:
fp16 does NOT work (ViT-g activations overflow, every output NaN). bf16 has
fp32's range and works; Turing emulates it, slower but correct. On Ampere+
(Colab A100/L4) bf16 is native.

Patches needed once weights are not fp32:
  1. Some submodules build fp32 tensors and feed them to Linear layers outside
     autocast. A global forward pre-hook casts fp32 args to the weight dtype.
  2. geometry.py builds its pose matrix in fp32; geometry and intrinsics
     recovery run in fp32.
  3. postprocess calls .numpy(), which has no bfloat16; outputs are upcast.
  4. With given intrinsics, ray dirs are cast to the weight dtype.
Do not pass use_amp=True on top of 16-bit weights.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import mapanything.models.mapanything.model as _model
import mapanything.utils.geometry as _geo
import mapanything.utils.inference as _inf
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

_patched = False


def _apply_patches(dtype):
    global _patched
    if _patched:
        return
    _patched = True

    def pre_hook(module, args):
        p = next(module.parameters(recurse=False), None)
        if p is None or p.dtype != dtype:
            return None
        changed, out = False, []
        for x in args:
            if torch.is_tensor(x) and x.dtype == torch.float32:
                x = x.to(dtype)
                changed = True
            out.append(x)
        return tuple(out) if changed else None

    torch.nn.modules.module.register_module_forward_pre_hook(pre_hook)

    _pm = _geo.convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap

    def fp32_pointmap(rd, dar, pt, pq):
        return _pm(rd.float(), dar.float(), pt.float(), pq.float()).to(rd.dtype)

    _geo.convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap = fp32_pointmap
    _model.convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap = fp32_pointmap

    _ri = _geo.recover_pinhole_intrinsics_from_ray_directions

    def fp32_intrinsics(rays, *a, **k):
        return _ri(rays.float(), *a, **k)

    _geo.recover_pinhole_intrinsics_from_ray_directions = fp32_intrinsics
    _inf.recover_pinhole_intrinsics_from_ray_directions = fp32_intrinsics

    _pp = _inf.postprocess_model_outputs_for_inference

    def up(x):
        if torch.is_tensor(x) and x.dtype in (torch.bfloat16, torch.float16):
            return x.float()
        if isinstance(x, dict):
            return {k: up(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(up(v) for v in x)
        return x

    def wrapped(raw_outputs, input_views, *a, **k):
        return _pp(up(raw_outputs), up(input_views), *a, **k)

    _inf.postprocess_model_outputs_for_inference = wrapped
    _model.postprocess_model_outputs_for_inference = wrapped

    _erd = MapAnything._encode_and_fuse_ray_dirs

    def ray_dirs_in_model_dtype(self, views, *a, **k):
        views = [{**v, "ray_directions_cam": v["ray_directions_cam"].to(dtype)}
                 if "ray_directions_cam" in v else v for v in views]
        return _erd(self, views, *a, **k)

    MapAnything._encode_and_fuse_ray_dirs = ray_dirs_in_model_dtype


def load_model(repo="facebook/map-anything-apache", dtype=torch.bfloat16):
    """Cast on CPU before moving to GPU so the fp32 copy never occupies VRAM."""
    if dtype != torch.float32:
        _apply_patches(dtype)
    model = MapAnything.from_pretrained(repo).eval().to(dtype).to("cuda")
    torch.cuda.empty_cache()
    return model


def infer(model, image_paths, longest_side=None):
    if longest_side:
        views = load_images(image_paths, resize_mode="longest_side", size=longest_side)
    else:
        views = load_images(image_paths)
    with torch.no_grad():
        return model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                           use_amp=False)
