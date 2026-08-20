"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import copy
import yaml
from typing import Any, Dict, List

from .workspace import GLOBAL_CONFIG

__all__ = [
    "load_config",
    "merge_config",
    "merge_dict",
    "parse_cli",
]


INCLUDE_KEY = "__include__"


def load_config(file_path, cfg=None):
    """Load a YAML file, recursively resolving optional ``__include__`` entries."""
    if cfg is None:
        cfg = {}
    _, ext = os.path.splitext(file_path)
    if ext not in (".yml", ".yaml"):
        raise ValueError(f"Only YAML configs are supported, got {file_path!r}.")

    with open(file_path, encoding="utf-8") as handle:
        file_cfg = yaml.safe_load(handle) or {}
    if not isinstance(file_cfg, dict):
        raise TypeError(f"Top-level YAML config must be a mapping: {file_path}")

    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg.pop(INCLUDE_KEY) or [])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)
            if not os.path.isabs(base_yaml):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)
            merge_dict(cfg, load_config(base_yaml, {}))

    return merge_dict(cfg, file_cfg)


def merge_dict(dct, another_dct, inplace=True) -> Dict:
    """merge another_dct into dct"""

    def _merge(dct, another) -> Dict:
        for k in another:
            if k in dct and isinstance(dct[k], dict) and isinstance(another[k], dict):
                _merge(dct[k], another[k])
            else:
                dct[k] = another[k]

        return dct

    if not inplace:
        dct = copy.deepcopy(dct)

    return _merge(dct, another_dct)


def dictify(s: str, v: Any) -> Dict:
    if "." not in s:
        return {s: v}
    key, rest = s.split(".", 1)
    return {key: dictify(rest, v)}


def parse_cli(nargs: List[str]) -> Dict:
    """
    parse command-line arguments
        convert `a.c=3 b=10` to `{'a': {'c': 3}, 'b': 10}`
    """
    cfg = {}
    if nargs is None or len(nargs) == 0:
        return cfg

    for s in nargs:
        s = s.strip()
        k, v = s.split("=", 1)
        d = dictify(k, yaml.safe_load(v))
        cfg = merge_dict(cfg, d)

    return cfg


def merge_config(cfg, another_cfg=GLOBAL_CONFIG, inplace: bool = False, overwrite: bool = False):
    """
    Merge another_cfg into cfg, return the merged config

    Example:

        cfg1 = load_config('./dfine_r18vd_6x_coco.yml')
        cfg1 = merge_config(cfg, inplace=True)

        cfg2 = load_config('./dfine_r50vd_6x_coco.yml')
        cfg2 = merge_config(cfg2, inplace=True)

        model1 = create(cfg1['model'], cfg1)
        model2 = create(cfg2['model'], cfg2)
    """

    def _merge(dct, another):
        for k in another:
            if k not in dct:
                dct[k] = another[k]

            elif isinstance(dct[k], dict) and isinstance(another[k], dict):
                _merge(dct[k], another[k])

            elif overwrite:
                dct[k] = another[k]

        return cfg

    if not inplace:
        cfg = copy.deepcopy(cfg)

    return _merge(cfg, another_cfg)
