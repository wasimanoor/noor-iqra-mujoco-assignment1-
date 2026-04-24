from __future__ import annotations
import re
from typing import Dict, Any, Optional


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s)
    return s


# You can expand this easily later
COLOR_ALIASES = {
    "red": "red_box",
    "green": "green_box",
    "blue": "blue_box",
    "yellow": "yellow_box",
    "white": "box",
    "gray": "box",
    "grey": "box",
}


TARGET_ALIASES = {
    "bin": "bin_center",
    "basket": "bin_center",
    "bucket": "bin_center",
    "left": "zone_left",
    "right": "zone_right",
    "zone left": "zone_left",
    "zone right": "zone_right",
}


def _extract_object_phrase(t: str) -> Optional[str]:
    """
    Extract object phrase after 'pick' or after 'where is' etc.
    Examples:
      pick red -> red
      pick green box -> green box
      pick red_box -> red_box
    """
    # pick <obj> (until place/stack/on/to/end)
    m = re.search(r"\bpick\b\s+(.+?)(?:\s+\b(place|stack|on|to)\b|$)", t)
    if m:
        return m.group(1).strip()
    # stack <obj> on <base>
    m = re.search(r"\bstack\b\s+(.+?)\s+\bon\b", t)
    if m:
        return m.group(1).strip()
    # where is <obj>
    m = re.search(r"\bwhere\s+is\b\s+(.+)$", t)
    if m:
        return m.group(1).strip()
    return None


def _resolve_obj_token(obj_phrase: Optional[str]) -> Optional[str]:
    if not obj_phrase:
        return None
    p = obj_phrase.strip().lower()
    p = p.replace("the ", "").strip()
    p = p.replace("cube", "box").strip()
    p = p.replace(" ", "_")  # "green box" -> "green_box"

    # direct color word
    if p in COLOR_ALIASES:
        return COLOR_ALIASES[p]

    # "green_box" etc
    if p in COLOR_ALIASES.values():
        return p

    # "green_box_something" not supported -> keep token
    return p


def parse_command(text: str) -> Dict[str, Any]:
    t = _normalize(text)

    if t in {"q", "quit", "exit"}:
        return {"task": "quit"}

    if t in {"h", "help", "?"}:
        return {"task": "help"}

    if t in {"reset", "home"}:
        return {"task": "reset"}

    if t in {"open", "open gripper", "gripper open"} or "open gripper" in t:
        return {"task": "gripper", "mode": "open"}

    if t in {"close", "close gripper", "gripper close"} or "close gripper" in t:
        return {"task": "gripper", "mode": "close"}

    if t in {"list", "list objects", "objects", "what objects"}:
        return {"task": "list_objects"}

    if t.startswith("where is"):
        obj = _resolve_obj_token(_extract_object_phrase(t))
        return {"task": "where", "obj": obj or "box"}

    # sort tasks
    if t in {"sort", "sort all", "sort everything"}:
        return {"task": "sort_all"}

    # make a tower
    if t in {"make a tower", "tower", "build a tower"}:
        return {"task": "tower"}

    # stack <obj> on <base>
    m_stack = re.search(r"\bstack\b\s+(.+?)\s+\bon\b\s+(.+)$", t)
    if m_stack:
        obj = _resolve_obj_token(m_stack.group(1))
        base = _resolve_obj_token(m_stack.group(2))
        return {"task": "stack", "obj": obj or "box", "base": base or "box"}

    # pick ... place ...
    if "pick" in t and "place" in t:
        obj = _resolve_obj_token(_extract_object_phrase(t)) or "box"

        # place on top: "place on green"
        m_on = re.search(r"\bplace\b\s+\bon\b\s+(.+)$", t)
        if m_on:
            base = _resolve_obj_token(m_on.group(1))
            return {"task": "place_on", "obj": obj, "base": base or "box"}

        # target alias (bin/left/right)
        for k, v in TARGET_ALIASES.items():
            if re.search(rf"\b{k}\b", t):
                return {"task": "pick_place", "obj": obj, "target": v}

        # x y numeric
        mx = re.search(r"\bx\s*(-?\d+(\.\d+)?)\b", t)
        my = re.search(r"\by\s*(-?\d+(\.\d+)?)\b", t)
        if mx and my:
            return {"task": "pick_place_xy", "obj": obj, "x": float(mx.group(1)), "y": float(my.group(1))}

        # default
        return {"task": "pick_place", "obj": obj, "target": "bin_center"}

    return {"task": "unknown", "text": text}
