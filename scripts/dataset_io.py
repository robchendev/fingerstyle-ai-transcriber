"""Shared file primitives for private preparation, releases and runtime outputs."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_json(path, value):
    path = Path(path).absolute()
    if path.resolve() != path or any(part.is_symlink() or part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400 for part in (path, *path.parents)) or path.exists() and path.stat().st_nlink > 1:
        raise ValueError(f"Output aliases are not allowed: {path}")
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
