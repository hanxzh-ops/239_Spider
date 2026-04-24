# src/config.py — thin shim; canonical config lives in assets/config.py
# Any module that does "from src.config import ..." still works.
from assets.config import *   # noqa: F401, F403
