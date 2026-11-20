"""Pull one number out of a free-form VLM answer and convert it to the task's unit.

Rules, in order:
  1. Drop anything inside <think>...</think> (reasoning models).
  2. If the text contains "final answer", parse only what follows the last one.
  3. A range ("2-3 m", "2 to 3 m") becomes its midpoint.
  4. Otherwise the first number wins.
An explicit unit next to the number is converted to the task's unit
(m, cm or m2). No unit means the number is taken as already in the task's unit,
since every VSI-Bench question names its unit.
"""

import re

# One capture group, so the "?" appended below makes the whole unit optional.
# (?![a-z]) instead of \b: \b fails after "²", which is not a word character.
_UNIT = (r"((?:square\s+met(?:er|re)s?|sq\.?\s*m|m\^?2|m²|square\s+feet|sq\.?\s*ft|ft\^?2|ft²"
         r"|centimet(?:er|re)s?|cm|millimet(?:er|re)s?|mm|met(?:er|re)s?|m"
         r"|feet|foot|ft|inch(?:es)?)(?![a-z]))")
_NUM = r"(\d+(?:\.\d+)?)"
_RANGE = re.compile(_NUM + r"\s*(?:-|–|to)\s*" + _NUM + r"\s*" + _UNIT + "?", re.I)
_SINGLE = re.compile(_NUM + r"\s*" + _UNIT + "?", re.I)

# factor to metres (length) or square metres (area)
_LEN = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "in": 0.0254}
_AREA = {"m2": 1.0, "ft2": 0.092903}


def _norm_unit(u):
    if not u:
        return None
    u = u.lower().replace(" ", "")
    if u.startswith(("square met", "squaremet", "sq.m", "sqm", "m^2", "m2", "m²")):
        return "m2"
    if u.startswith(("squarefeet", "sq.ft", "sqft", "ft^2", "ft2", "ft²")):
        return "ft2"
    if u.startswith("centimet") or u == "cm":
        return "cm"
    if u.startswith("millimet") or u == "mm":
        return "mm"
    if u.startswith("met") or u == "m":
        return "m"
    if u in ("feet", "foot", "ft"):
        return "ft"
    if u.startswith("inch"):
        return "in"
    return None


def _clean(text):
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)   # 1,200 -> 1200
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)        # 1,5 -> 1.5
    low = text.lower()
    i = low.rfind("final answer")
    return text[i + len("final answer"):] if i >= 0 else text


def convert(value, unit, target):
    """Convert value in `unit` to `target` (m, cm, m2). Unknown combos pass through."""
    if unit is None or unit == target:
        return value
    if target in ("m", "cm") and unit in _LEN:
        metres = value * _LEN[unit]
        return metres if target == "m" else metres * 100.0
    if target == "m2" and unit in _AREA:
        return value * _AREA[unit]
    # e.g. "3.5 m" for a room size question: the model dropped the "squared".
    return value


def parse_answer(text, target_unit):
    """Return (value or None, unit_seen or None)."""
    if text is None:
        return None, None
    t = _clean(text)
    m = _RANGE.search(t)
    s = _SINGLE.search(t)
    if m and (not s or m.start() <= s.start()):
        v = (float(m.group(1)) + float(m.group(2))) / 2.0
        u = _norm_unit(m.group(3))
    elif s:
        v = float(s.group(1))
        u = _norm_unit(s.group(2))
    else:
        return None, None
    return convert(v, u, target_unit), u
