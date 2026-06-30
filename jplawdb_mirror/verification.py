from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .core import Config, TEXT_EXTENSIONS, safe_relative_path, validate_metrics


class VerificationError(RuntimeError):
    """The generated mirror is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    file_count: int
    total_bytes: int
    html_links_checked: int


class _HTMLReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name.lower() in {"href", "src"} and value:
                self.references.append(value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read manifest: {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        raise VerificationError("manifest schema_version must be 2")
    if not isinstance(data.get("files"), list):
        raise VerificationError("manifest files must be a list")
    if not isinstance(data.get("metrics"), dict):
        raise VerificationError("manifest metrics must be a mapping")
    return data


def _manifest_entries(manifest: dict[str, Any]) -> dict[str, tuple[int, str]]:
    entries: dict[str, tuple[int, str]] = {}
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise VerificationError("manifest contains a non-object file entry")
        raw_path = item.get("path")
        raw_size = item.get("size")
        raw_hash = item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(raw_size, int) or raw_size < 0:
            raise VerificationError(f"invalid manifest entry: {item!r}")
        if not isinstance(raw_hash, str) or len(raw_hash) != 64:
            raise VerificationError(f"invalid SHA-256 in manifest: {raw_path}")
        try:
            path = safe_relative_path(raw_path)
        except ValueError as exc:
            raise VerificationError(f"unsafe manifest path: {raw_path}") from exc
        if path in entries:
            raise VerificationError(f"duplicate manifest path: {path}")
        entries[path] = (raw_size, raw_hash.lower())
    return entries


def _actual_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }


def _local_reference(source_path: str, reference: str) -> str | None:
    if not reference or reference.startswith(("#", "//")):
        return None
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc:
        return None
    if parsed.path.startswith(("data:", "javascript:", "mailto:", "tel:")):
        return None
    raw_path = unquote(parsed.path).replace("\\", "/")
    if not raw_path:
        return None
    if raw_path.startswith("/"):
        candidate = raw_path.lstrip("/")
    else:
        candidate = posixpath.join(posixpath.dirname(source_path), raw_path)
    candidate = posixpath.normpath(candidate)
    if candidate in {"", "."}:
        candidate = source_path
    if candidate == ".." or candidate.startswith("../"):
        return f"!ESCAPES!{candidate}"
    if raw_path.endswith("/"):
        candidate = posixpath.join(candidate, "index.html")
    return candidate


def _check_html_links(root: Path, files: dict[str, Path], errors: list[str]) -> int:
    checked = 0
    for relative, path in sorted(files.items()):
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"HTML is not UTF-8: {relative}")
            continue
        parser = _HTMLReferenceParser()
        try:
            parser.feed(text)
        except Exception as exc:  # HTMLParser is tolerant, but malformed state must be visible.
            errors.append(f"cannot parse HTML {relative}: {exc}")
            continue
        for reference in parser.references:
            target = _local_reference(relative, reference)
            if target is None:
                continue
            checked += 1
            if target.startswith("!ESCAPES!"):
                errors.append(f"link escapes mirror root: {relative} -> {reference}")
                continue
            if target not in files:
                errors.append(f"broken local link: {relative} -> {reference} ({target})")
    return checked


def _check_egov_sync(
    actual: dict[str, Path],
    manifest: dict[str, Any],
    config: Config,
    errors: list[str],
) -> None:
    if not config.egov_laws:
        return
    from .egov import JST, resolve_egov_as_of

    required = {
        "egov-law-db/index.html",
        "egov-law-db/index.json",
        "egov-law-db/llms.txt",
        "egov-law-db/quickstart.txt",
        "egov-law-db/status.json",
    }
    errors.extend(
        f"required e-Gov file is missing: {path}"
        for path in sorted(required - actual.keys())
    )
    status_path = actual.get("egov-law-db/status.json")
    if status_path is None:
        return
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read e-Gov status: {exc}")
        return
    if not isinstance(status, dict) or status.get("schema_version") != 1:
        errors.append("e-Gov status schema_version must be 1")
        return
    actual_as_of = status.get("as_of")
    if config.egov_as_of == "current":
        try:
            actual_date = date.fromisoformat(str(actual_as_of))
            today = datetime.now(JST).date()
            age_days = (today - actual_date).days
            if age_days < 0 or age_days > 1:
                errors.append(
                    f"e-Gov status is stale: current={today} actual={actual_as_of}"
                )
        except ValueError:
            errors.append(f"invalid e-Gov status as_of: {actual_as_of}")
    else:
        expected_as_of = resolve_egov_as_of(config.egov_as_of)
        if actual_as_of != expected_as_of:
            errors.append(
                f"e-Gov status date mismatch: expected={expected_as_of} actual={actual_as_of}"
            )
    laws = status.get("laws")
    if not isinstance(laws, list):
        errors.append("e-Gov status laws must be a list")
        return
    expected_codes = set(config.egov_laws)
    actual_codes = {
        str(item.get("code"))
        for item in laws
        if isinstance(item, dict) and isinstance(item.get("code"), str)
    }
    if actual_codes != expected_codes:
        errors.append(
            f"e-Gov law codes mismatch: missing={sorted(expected_codes - actual_codes)} "
            f"extra={sorted(actual_codes - expected_codes)}"
        )
    article_total = 0
    for item in laws:
        if not isinstance(item, dict):
            errors.append("e-Gov status contains a non-object law")
            continue
        code = item.get("code")
        if not isinstance(code, str) or code not in config.egov_laws:
            continue
        for field in ("law_id", "law_num", "law_revision_id", "xml_sha256"):
            if not isinstance(item.get(field), str) or not item[field]:
                errors.append(f"e-Gov metadata is missing {field}: {code}")
        article_count = item.get("article_count")
        if not isinstance(article_count, int) or article_count < 1:
            errors.append(f"invalid e-Gov article count: {code}")
            continue
        article_total += article_count
        xml_path = f"egov-law-db/xml/{code}.xml"
        metadata_path = f"egov-law-db/metadata/{code}.json"
        index_path = f"egov-law-db/text/{code}/index.html"
        for path in (xml_path, metadata_path, index_path):
            if path not in actual:
                errors.append(f"required e-Gov law file is missing: {path}")
        text_prefix = f"egov-law-db/text/{code}/"
        text_count = sum(
            1
            for path in actual
            if path.startswith(text_prefix) and path.endswith(".txt")
        )
        if text_count != article_count:
            errors.append(
                f"e-Gov article files mismatch for {code}: "
                f"expected={article_count} actual={text_count}"
            )
    if status.get("law_count") != len(config.egov_laws):
        errors.append(
            f"e-Gov status law_count mismatch: "
            f"expected={len(config.egov_laws)} actual={status.get('law_count')}"
        )
    if status.get("article_count") != article_total:
        errors.append(
            f"e-Gov status article_count mismatch: "
            f"declared={status.get('article_count')} actual={article_total}"
        )
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict):
        if metrics.get("egov_law_codes") != len(config.egov_laws):
            errors.append("manifest e-Gov law count is inconsistent")
        if metrics.get("egov_main_articles") != article_total:
            errors.append("manifest e-Gov article count is inconsistent")


def verify_output(root: Path, config: Config) -> VerificationReport:
    root = root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        raise VerificationError(f"output directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise VerificationError(f"manifest does not exist: {manifest_path}")
    for required in ("index.html", ".nojekyll", "download_log.tsv"):
        if not (root / required).is_file():
            errors.append(f"required generated file is missing: {required}")

    manifest = _load_manifest(manifest_path)
    entries = _manifest_entries(manifest)
    actual = _actual_files(root)

    missing = sorted(set(entries) - set(actual))
    extra = sorted(set(actual) - set(entries))
    errors.extend(f"manifest file is missing: {path}" for path in missing)
    errors.extend(f"file is absent from manifest: {path}" for path in extra)

    total_bytes = 0
    for relative in sorted(set(entries) & set(actual)):
        expected_size, expected_hash = entries[relative]
        path = actual[relative]
        size = path.stat().st_size
        total_bytes += size
        if size != expected_size:
            errors.append(
                f"size mismatch: {relative} expected={expected_size} actual={size}"
            )
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch: {relative}")

    source_marker = config.source_base.encode("utf-8")
    for relative, path in sorted(actual.items()):
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        data = path.read_bytes()
        if source_marker in data:
            errors.append(f"source URL was not rewritten: {relative}")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"text file is not UTF-8: {relative}")

    metrics_raw = manifest.get("metrics")
    try:
        metrics = {str(k): int(v) for k, v in metrics_raw.items()}
        validate_metrics(metrics, config.minimum_counts)
    except (TypeError, ValueError, RuntimeError) as exc:
        errors.append(str(exc))

    _check_egov_sync(actual, manifest, config, errors)
    link_targets = dict(actual)
    link_targets["manifest.json"] = manifest_path
    checked_links = _check_html_links(root, link_targets, errors)
    if errors:
        preview = errors[:100]
        remainder = len(errors) - len(preview)
        message = "mirror verification failed:\n- " + "\n- ".join(preview)
        if remainder:
            message += f"\n- ... and {remainder} more error(s)"
        raise VerificationError(message)
    return VerificationReport(
        file_count=len(actual),
        total_bytes=total_bytes,
        html_links_checked=checked_links,
    )
