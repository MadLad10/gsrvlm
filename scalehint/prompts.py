"""Conditions and prompt text.

Condition names:
  baseline                       no hint
  placebo                        hint line with no number
  spatialprompt                  reference-object reasoning instruction (Liao et al. 2024 style)
  teacher[_k<k>]                 MapAnything median depth, optionally multiplied by k
  teacher_ext                    teacher line + scene extent (for room size)
  shuffled                       teacher value from a different scene (derangement)
  learned                        predicted scale from predictor/train.py
  anchor[_k<k>]                  ground-truth size of a different object in the same scene, times k
"""

import re

POSTFIX = "Please answer the question using a single word or phrase."
POSTFIX_REASON = ("Think step by step, then end your reply with a line of the form "
                  "'Final answer: <number>'.")

PLACEBO = "Scale reference: consider the physical scale of the scene carefully."
SPATIAL = ("Before answering, pick one or more reference objects in the video whose "
           "real-world size you know well, estimate their size, and use them as a "
           "ruler to measure what is asked.")

_COND = re.compile(r"(baseline|placebo|spatialprompt|teacher_ext|teacher|shuffled|learned|anchor|answer)"
                   r"(?:_k(\d+(?:\.\d+)?))?$")

K_LEVELS = [0.5, 0.71, 1.41, 2.0]


def parse_condition(name):
    m = _COND.match(name)
    if not m:
        raise ValueError(f"unknown condition {name!r}")
    src = m.group(1)
    k = float(m.group(2)) if m.group(2) else 1.0
    if m.group(2) and src not in ("teacher", "anchor", "learned", "answer"):
        raise ValueError(f"_k only applies to teacher/anchor/learned/answer, got {name!r}")
    return src, k


def depth_line(x):
    return (f"Scale reference: the typical distance from the camera to visible "
            f"surfaces in this video is about {x:.1f} m.")


def anchor_line(obj, cm):
    return (f"Scale reference: the {obj} in this video is about {cm:.0f} cm along its "
            f"longest dimension.")


UNIT_WORDS = {"m": "m", "cm": "cm", "m2": "square meters"}


def answer_line(x, unit):
    """Positive control: the ground-truth answer (times k) stated directly."""
    fmt = f"{x:.0f}" if unit == "cm" else f"{x:.1f}"
    return f"Hint: the correct answer is approximately {fmt} {UNIT_WORDS[unit]}."


def build_prompt(question, src, hint=None, extent=None, anchor_obj=None, unit=None):
    """Returns (prompt, reasoning?)."""
    lines = []
    if src == "answer":
        lines.append(answer_line(hint, unit))
    elif src in ("teacher", "shuffled", "learned"):
        lines.append(depth_line(hint))
    elif src == "teacher_ext":
        lines.append(depth_line(hint) + f" The scene spans about {extent:.1f} m.")
    elif src == "placebo":
        lines.append(PLACEBO)
    elif src == "anchor":
        lines.append(anchor_line(anchor_obj, hint))
    elif src == "spatialprompt":
        lines.append(SPATIAL)
    lines.append(question)
    reasoning = src == "spatialprompt"
    lines.append(POSTFIX_REASON if reasoning else POSTFIX)
    return "\n".join(lines), reasoning
