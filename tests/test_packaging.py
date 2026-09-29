"""Verify that research code is usable independently of the checkout layout."""

import json
import subprocess
import sys


def test_installed_package_imports_outside_checkout(tmp_path):
    probe = """
import importlib.metadata
import json
import shouldyouevennn
print(json.dumps({
    "package": shouldyouevennn.__name__,
    "distribution": importlib.metadata.metadata("shouldyouevennn")["Name"],
}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "package": "shouldyouevennn",
        "distribution": "shouldyouevennn",
    }


def test_distribution_excludes_historical_packages(tmp_path):
    probe = """
import importlib.util
import json
print(json.dumps({
    name: importlib.util.find_spec(name) is not None
    for name in ("ebe", "forecaster", "candidates", "architecture_generator")
}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not any(json.loads(result.stdout).values())
