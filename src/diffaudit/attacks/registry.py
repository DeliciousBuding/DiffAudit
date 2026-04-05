"""Attack planner registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from diffaudit.attacks.clid import build_clid_plan
from diffaudit.attacks.pia import build_pia_plan
from diffaudit.attacks.secmi import build_secmi_plan


Planner = Callable[[Any], Any]


def get_attack_planner(method: str) -> Planner:
    if method == "secmi":
        return build_secmi_plan
    if method == "pia":
        return build_pia_plan
    if method == "clid":
        return build_clid_plan
    raise ValueError(f"Unsupported attack method: {method}")
