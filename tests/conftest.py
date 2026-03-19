"""
conftest.py — shared fixtures for physics tests.
Forces garbage collection between test classes to release
PyBullet DIRECT connections that have a limited pool.
"""

import gc
import pytest


@pytest.fixture(autouse=True)
def _cleanup_pybullet():
    yield
    gc.collect()
