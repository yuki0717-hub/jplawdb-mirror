#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

OUTPUT_DIR = Path("output")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TARGET_URL = "jplawdb.github.io"


def load_manifest(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries: dict[str, int] = {}

    def put_entry(raw_path: Any, raw_size: Any) -> None:
        if not isinstance(raw_path, str):
            return
        try:
            size = int(raw_size)
        except (TypeError, ValueError):
            return
        norm = raw_path.replace("\\", "/").lstrip("./")
        entries[norm] = size

    if isinstance(data, dict):
        if "files" in data and isinstance(data["files"], list):
            for item in data["files"]:
                if isinstance(item, dict):
                    put_entry(item.get("path") or item.get("file"), item.get("size") or item.get("bytes"))
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    put_entry(k, v.get("size") or v.get("bytes"))
                else:
                    put_entry(k, v)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                put_entry(item.get("path") or item.get("file"), item.get("size") or item.get("bytes"))

    return entries


def iter_output_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path != MANIFEST_PATH:
            yield path


def detect_url(path: Path, target: str) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return target in f.read()
    except OSError:
        return False


def main() -> int:
    ng_count = 0
    ok_count = 0
    remain_urls: list[str] = []

    if not OUTPUT_DIR.exists() or not OUTPUT_DIR.is_dir():
        print(f"[NG] output directory not found: {OUTPUT_DIR}")
        print("OK: 0")
        print("NG: 1")
        print("Residual URL files:")
        return 1

    if not MANIFEST_PATH.exists():
        print(f"[NG] manifest not found: {MANIFEST_PATH}")
        print("OK: 0")
        print("NG: 1")
        print("Residual URL files:")
        return 1

    try:
        manifest_entries = load_manifest(MANIFEST_PATH)
    except Exception as e:  # noqa: BLE001
        print(f"[NG] failed to load manifest: {e}")
        print("OK: 0")
        print("NG: 1")
        print("Residual URL files:")
        return 1

    actual_entries: dict[str, int] = {}
    for file_path in iter_output_files(OUTPUT_DIR):
        rel = file_path.relative_to(OUTPUT_DIR).as_posix()
        actual_entries[rel] = file_path.stat().st_size
        if detect_url(file_path, TARGET_URL):
            remain_urls.append(rel)

    for rel, expected_size in manifest_entries.items():
        if rel not in actual_entries:
            ng_count += 1
            print(f"[NG] missing file: {rel}")
            continue

        actual_size = actual_entries[rel]
        if actual_size != expected_size:
            ng_count += 1
            print(f"[NG] size mismatch: {rel} (expected={expected_size}, actual={actual_size})")
        else:
            ok_count += 1

    extra_files = sorted(set(actual_entries) - set(manifest_entries))
    for rel in extra_files:
        ng_count += 1
        print(f"[NG] extra file not in manifest: {rel}")

    if remain_urls:
        ng_count += len(remain_urls)
        print(f"[NG] residual URL '{TARGET_URL}' found in {len(remain_urls)} file(s)")
    else:
        print(f"[OK] no residual URL '{TARGET_URL}'")

    print(f"OK: {ok_count}")
    print(f"NG: {ng_count}")
    print("Residual URL files:")
    for rel in sorted(remain_urls):
        print(f"- {rel}")

    return 1 if ng_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
