"""Shared test fixtures."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FW = Path(__file__).resolve().parent.parent / "firmware" / "leg-esp32"


@pytest.fixture(scope="session")
def host(tmp_path_factory):
    """The leg firmware's pure parts compiled for the PC (firmware/leg-esp32/host_test/host_check.c): run(lines)
    feeds it commands and returns its output lines (tests/test_firmware_host.py, tests/test_sts.py)."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler on the PATH")
    exe = tmp_path_factory.mktemp("fw") / ("host_check.exe" if sys.platform == "win32" else "host_check")
    src = [FW / "host_test" / "host_check.c", FW / "main" / "sts.c", FW / "main" / "leg.c", FW / "main" / "xmath.c"]
    subprocess.run([cc, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Werror", "-o", str(exe),
                    *map(str, src), "-lm"], check=True)

    def run(lines):
        out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
        return out.stdout.splitlines()

    return run
