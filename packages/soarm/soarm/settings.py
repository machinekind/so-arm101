"""Strict dataclass builder for the backends' config sections (`feetech:`, `lerobot:`).

`from_mapping(cls, data)` builds the settings dataclass `cls` from a mapping, such as a section of the
twin's config file. Unknown keys and missing required keys fail; the other fields take their defaults.
Values are cast to the field types: tuples, dicts, nested dataclasses and paths.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints


def _coerce(value: Any, target_type: Any, where: str) -> Any:
    """Casts a YAML value to the dataclass field type: tuples, nested mappings, dataclasses."""
    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin in (Union, types.UnionType):
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], where) if len(inner) == 1 else value
    if is_dataclass(target_type):
        return from_mapping(target_type, value, where)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], where) for v in value)
        if args and len(value) != len(args):
            raise ValueError(f"{where}: oczekiwano {len(args)} wartosci, dostalem {len(value)}")
        return tuple(_coerce(v, t, where) for v, t in zip(value, args)) if args else tuple(value)
    if origin is dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}: oczekiwano mapowania, dostalem {type(value).__name__}")
        _, val_t = args
        return {k: _coerce(v, val_t, f"{where}.{k}") for k, v in value.items()}
    if target_type is Path:
        return Path(value)
    return value


def from_mapping(cls: Any, data: Any, where: str | None = None) -> Any:
    """Builds the dataclass `cls` from a mapping, strictly: unknown keys fail, gaps take defaults.

    `where` names the section in error messages (default: the class name).
    """
    where = where or cls.__name__
    if not isinstance(data, Mapping):
        raise TypeError(f"Oczekiwano mapowania dla {where}, dostalem {type(data).__name__}")

    # `from __future__ import annotations` turns annotations into strings, so resolve them to
    # real types before casting anything.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Nieznane klucze w sekcji {where}: {sorted(unknown)}")
    missing = [f.name for f in fields(cls)
               if f.name not in data and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ValueError(f"Brak kluczy w sekcji {where}: {missing}")

    kwargs = {name: _coerce(value, hints[name], f"{where}.{name}") for name, value in data.items()}
    return cls(**kwargs)
