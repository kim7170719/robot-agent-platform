#!/usr/bin/env sh
set -e
cd "$(dirname "$0")/.."
pip install -U pip
pip install -e ".[dev]"
