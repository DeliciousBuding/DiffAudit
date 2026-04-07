from __future__ import annotations

import json
import os
import platform
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso_local() -> str:
    # Local time with explicit offset for provenance.
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def norm_abs(p: str | Path) -> str:
    return str(Path(p).expanduser().resolve())


def is_within(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def json_dumps_canonical(obj: Any) -> bytes:
    # Stable encoding for hashing: sorted keys, no insignificant whitespace.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def host_info() -> dict[str, Any]:
    u = platform.uname()
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": sys.platform,
        "python": sys.version.replace("\n", " "),
        "uname": {
            "system": u.system,
            "node": u.node,
            "release": u.release,
            "version": u.version,
            "machine": u.machine,
            "processor": u.processor,
        },
        "pid": os.getpid(),
        "cwd": str(Path.cwd().resolve()),
    }


@dataclass
class CheckResult:
    ok: bool
    errors: list[str]
    warnings: list[str]

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}

