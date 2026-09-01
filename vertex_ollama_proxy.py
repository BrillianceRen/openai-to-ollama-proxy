#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vertex_ollama_proxy.py
Python importable module wrapper for vertex-ollama-proxy.py
"""
import importlib.util
import os
import sys

_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vertex-ollama-proxy.py")
_spec = importlib.util.spec_from_file_location("vertex_ollama_proxy_impl", _script_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["vertex_ollama_proxy"] = _mod
_spec.loader.exec_module(_mod)

# Re-export all public attributes
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v

if __name__ == "__main__":
    _mod.main()
