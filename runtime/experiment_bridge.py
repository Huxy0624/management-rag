from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_scripts_on_path() -> None:
    scripts_path = str(SCRIPTS_DIR)
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)


@lru_cache(maxsize=1)
def get_v1():
    ensure_scripts_on_path()
    return importlib.import_module("exp_generate_answers_v1")


@lru_cache(maxsize=1)
def get_v2():
    ensure_scripts_on_path()
    return importlib.import_module("exp_generate_answers_v2")


@lru_cache(maxsize=1)
def get_v21():
    ensure_scripts_on_path()
    return importlib.import_module("exp_generate_answers_v21")
