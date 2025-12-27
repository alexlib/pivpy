from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _dist_version
from typing import Optional


@dataclass(frozen=True)
class UpdateCheckResult:
    """Result of a package update check against PyPI."""

    status: int
    installed: str
    latest: str


def _get_installed_version_str(dist_name: str = "pivpy") -> str:
    try:
        return _dist_version(dist_name)
    except PackageNotFoundError:
        return "0.0.0"


def _get_pypi_version_str(package: str = "pivpy", timeout: float = 3.0) -> str:
    url = f"https://pypi.org/pypi/{package}/json"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "pivpy-update-check/1.0 (+https://pypi.org/project/pivpy/)",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()

    data = json.loads(payload.decode("utf-8"))
    return str(data["info"]["version"])


def _compare_versions(installed: str, latest: str) -> int:
    """Compare version strings.

    Returns:
        -1 if installed < latest
         0 if installed == latest
         1 if installed > latest

    Prefers packaging.version.Version when available; falls back to a simple
    numeric comparison for dotted versions.
    """

    try:
        from packaging.version import Version  # type: ignore

        a = Version(installed)
        b = Version(latest)
        return (a > b) - (a < b)
    except Exception:
        # Fallback: compare numeric components only (best-effort).
        def parts(v: str) -> tuple[int, ...]:
            nums = re.findall(r"\d+", v)
            if not nums:
                return (0,)
            return tuple(int(n) for n in nums)

        a = parts(installed)
        b = parts(latest)
        return (a > b) - (a < b)


def check_update(
    package: str = "pivpy",
    *,
    dist_name: Optional[str] = None,
    timeout: float = 3.0,
    verbose: bool = False,
) -> UpdateCheckResult:
    """Check whether a newer version of pivpy exists on PyPI.

    Status codes (mirrors the PIVMat convention):
        0: server unavailable / request failed
        1: no new version available (installed == latest)
        2: a new version is available online (installed < latest)
        3: the online version is older than installed (installed > latest)

    Args:
        package: PyPI package name to query (defaults to "pivpy").
        dist_name: Installed distribution name (defaults to package).
        timeout: Network timeout (seconds).
        verbose: If True, prints a short human-readable message.

    Returns:
        UpdateCheckResult with status + installed/latest strings.
    """

    dist = dist_name or package
    installed = _get_installed_version_str(dist)

    try:
        latest = _get_pypi_version_str(package, timeout=timeout)
    except Exception:
        if verbose:
            print("Update check failed: server unavailable.")
        return UpdateCheckResult(status=0, installed=installed, latest="")

    cmp = _compare_versions(installed, latest)
    if cmp < 0:
        status = 2
        if verbose:
            print(f"New version available: {package} {latest} (installed: {installed}).")
    elif cmp > 0:
        status = 3
        if verbose:
            print(f"Installed version ({installed}) is newer than PyPI ({latest}).")
    else:
        status = 1
        if verbose:
            print(f"{package} is up to date ({installed}).")

    return UpdateCheckResult(status=status, installed=installed, latest=latest)
