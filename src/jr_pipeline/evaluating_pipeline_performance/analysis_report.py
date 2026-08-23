"""Text rendering for inspection output. Keep it boring; structure wins."""
from __future__ import annotations

import json
from typing import Any


def pretty(obj: Any) -> str:
    """Indented, sort-keyed JSON string for human-readable inspection output."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
