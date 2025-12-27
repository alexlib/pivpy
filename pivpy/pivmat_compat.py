from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class _BracketSpec:
    prefix: str
    expr: str
    fmt: str
    suffix: str


def _parse_range_expr(expr: str) -> list[float]:
    """Parse a restricted MATLAB-like range expression.

    Supported forms:
    - "1:5" (inclusive)
    - "1:0.5:2" (inclusive end if close)
    - "1 2 3" or "1,2,3"

    This intentionally does NOT eval arbitrary code.
    """

    s = expr.strip()
    if not s:
        return []

    # Colon form
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) == 2:
            start, end = (float(parts[0]), float(parts[1]))
            step = 1.0 if end >= start else -1.0
        elif len(parts) == 3:
            start, step, end = (float(parts[0]), float(parts[1]), float(parts[2]))
            if step == 0:
                raise ValueError("Invalid range: step must be nonzero")
        else:
            raise ValueError(f"Invalid range expression: {expr!r}")

        # Generate inclusive range, tolerant to float error.
        out: list[float] = []
        cur = start
        # direction-aware loop
        if step > 0:
            while cur <= end + 1e-12:
                out.append(cur)
                cur += step
        else:
            while cur >= end - 1e-12:
                out.append(cur)
                cur += step
        return out

    # List form (spaces/commas)
    toks = [t for t in re.split(r"[\s,]+", s) if t]
    return [float(t) for t in toks]


def _split_first_bracket(s: str) -> _BracketSpec | None:
    p1 = s.find("[")
    if p1 < 0:
        return None
    p2 = s.find("]", p1 + 1)
    if p2 < 0:
        raise ValueError("Invalid string: Missing closing bracket ']'")

    prefix = s[:p1]
    inside = s[p1 + 1 : p2]
    suffix = s[p2 + 1 :]

    # Optional ",NZ" where NZ can be "5" or "2.3" (width.precision)
    if "," in inside:
        expr, nz = inside.split(",", 1)
        nz = nz.strip()
        try:
            nz_val = float(nz)
        except ValueError as e:
            raise ValueError(f"Invalid NZ format specifier: {nz!r}") from e
    else:
        expr, nz_val = inside, 5.0

    width = int(nz_val)
    prec = int(round(10 * (nz_val - int(nz_val))))
    if width > 16:
        raise ValueError("Invalid number of zero padding: too large")
    fmt = f"{{:0{width}.{prec}f}}"

    return _BracketSpec(prefix=prefix, expr=expr.strip(), fmt=fmt, suffix=suffix)


def expandstr(pattern: str) -> list[str]:
    """Expand indexed bracket strings (PIVMAT-compatible subset).

    Port of PIVMAT's ``expandstr.m`` with a safety constraint: only a restricted
    range grammar is supported (no arbitrary eval).

    Examples
    --------
    - ``expandstr('DSC[2:2:8,4].JPG')`` -> ['DSC0002.JPG', ...]
    - ``expandstr('dt=[1:0.5:2,2.3]s')`` -> ['dt=1.000s', ...]
    - Multiple brackets are expanded recursively.
    """

    spec = _split_first_bracket(str(pattern))
    if spec is None:
        return [str(pattern)]

    nums = _parse_range_expr(spec.expr)
    out = [f"{spec.prefix}{spec.fmt.format(v)}{spec.suffix}" for v in nums]

    # Recurse if there are remaining brackets in the suffix.
    if "[" in spec.suffix:
        expanded: list[str] = []
        for s in out:
            expanded.extend(expandstr(s))
        return expanded

    return out
