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
    supplementary_total = 0
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
        supplementary_count = item.get("supplementary_count")
        if not isinstance(supplementary_count, int) or supplementary_count < 0:
            errors.append(f"invalid e-Gov supplementary count: {code}")
            supplementary_count = 0
        supplementary_total += supplementary_count
        xml_path = f"egov-law-db/xml/{code}.xml"
        metadata_path = f"egov-law-db/metadata/{code}.json"
        index_path = f"egov-law-db/text/{code}/index.html"
        supplementary_path = f"egov-law-db/supplementary/{code}.txt"
        for path in (xml_path, metadata_path, index_path, supplementary_path):
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
    if status.get("supplementary_count") != supplementary_total:
        errors.append(
            f"e-Gov status supplementary_count mismatch: "
            f"declared={status.get('supplementary_count')} actual={supplementary_total}"
        )
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict):
        if metrics.get("egov_law_codes") != len(config.egov_laws):
            errors.append("manifest e-Gov law count is inconsistent")
        if metrics.get("egov_main_articles") != article_total:
            errors.append("manifest e-Gov article count is inconsistent")
        if metrics.get("egov_supplementary_provisions") != supplementary_total:
            errors.append("manifest e-Gov supplementary count is inconsistent")


def _check_nta_sync(
    actual: dict[str, Path],
    manifest: dict[str, Any],
    config: Config,
    errors: list[str],
) -> None:
    if not config.nta_sources:
        return
    from .nta import JST, _match_text

    required = {
        "nta-official-db/index.html",
        "nta-official-db/index.json",
        "nta-official-db/quickstart.txt",
        "nta-official-db/status.json",
    }
    errors.extend(
        f"required NTA official file is missing: {path}"
        for path in sorted(required - actual.keys())
    )
    status_path = actual.get("nta-official-db/status.json")
    if status_path is None:
        return
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read NTA official status: {exc}")
        return
    if not isinstance(status, dict) or status.get("schema_version") != 1:
        errors.append("NTA official status schema_version must be 1")
        return
    try:
        fetched_on = date.fromisoformat(str(status.get("fetched_on")))
        age_days = (datetime.now(JST).date() - fetched_on).days
        if age_days < 0 or age_days > 1:
            errors.append(
                f"NTA official status is stale: "
                f"current={datetime.now(JST).date()} fetched_on={fetched_on}"
            )
    except ValueError:
        errors.append(f"invalid NTA official fetched_on: {status.get('fetched_on')}")
    sources = status.get("sources")
    if not isinstance(sources, list):
        errors.append("NTA official status sources must be a list")
        return
    expected_codes = set(config.nta_sources)
    actual_codes = {
        str(item.get("code"))
        for item in sources
        if isinstance(item, dict) and isinstance(item.get("code"), str)
    }
    if actual_codes != expected_codes:
        errors.append(
            f"NTA official source codes mismatch: "
            f"missing={sorted(expected_codes - actual_codes)} "
            f"extra={sorted(actual_codes - expected_codes)}"
        )
    for item in sources:
        if not isinstance(item, dict):
            errors.append("NTA official status contains a non-object source")
            continue
        code = item.get("code")
        if not isinstance(code, str) or code not in config.nta_sources:
            continue
        spec = config.nta_sources[code]
        if item.get("source_url") != spec.url:
            errors.append(f"NTA official source URL mismatch: {code}")
        if _match_text(spec.title) not in _match_text(str(item.get("title") or "")):
            errors.append(f"NTA official source title mismatch: {code}")
        text_relative = f"nta-official-db/{item.get('text_path')}"
        raw_relative = f"nta-official-db/{item.get('raw_path')}"
        metadata_relative = f"nta-official-db/metadata/{code}.json"
        for relative in (text_relative, raw_relative, metadata_relative):
            if relative not in actual:
                errors.append(f"required NTA official source file is missing: {relative}")
        text_path = actual.get(text_relative)
        if text_path is not None:
            if item.get("text_sha256") != _sha256(text_path):
                errors.append(f"NTA official text SHA-256 mismatch: {code}")
            text = text_path.read_text(encoding="utf-8")
            normalized_text = _match_text(text)
            missing_terms = [
                term
                for term in spec.required_terms
                if _match_text(term) not in normalized_text
            ]
            if missing_terms:
                errors.append(
                    f"NTA official text lost required terms: {code}: {missing_terms}"
                )
        raw_path = actual.get(raw_relative)
        if raw_path is not None and item.get("raw_sha256") != _sha256(raw_path):
            errors.append(f"NTA official raw SHA-256 mismatch: {code}")
        legal_as_of = item.get("legal_as_of")
        if spec.require_legal_date and not isinstance(legal_as_of, str):
            errors.append(f"NTA official legal date is missing: {code}")
        if isinstance(legal_as_of, str):
            try:
                legal_date = date.fromisoformat(legal_as_of)
                legal_age = (datetime.now(JST).date() - legal_date).days
                if legal_age < 0 or legal_age > config.nta_max_legal_age_days:
                    errors.append(
                        f"NTA official legal date is stale: {code}: age_days={legal_age}"
                    )
            except ValueError:
                errors.append(f"NTA official legal date is invalid: {code}: {legal_as_of}")
    if status.get("source_count") != len(config.nta_sources):
        errors.append(
            f"NTA official source_count mismatch: "
            f"expected={len(config.nta_sources)} actual={status.get('source_count')}"
        )
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict):
        if metrics.get("nta_official_documents") != len(config.nta_sources):
            errors.append("manifest NTA official source count is inconsistent")


def _check_egov_history(
    actual: dict[str, Path],
    manifest: dict[str, Any],
    config: Config,
    errors: list[str],
) -> None:
    if not config.egov_history_dates:
        return
    required = {
        "egov-law-db/history/index.html",
        "egov-law-db/history/index.json",
        "egov-law-db/history/quickstart.txt",
        "egov-law-db/history/status.json",
    }
    errors.extend(
        f"required e-Gov history file is missing: {path}"
        for path in sorted(required - actual.keys())
    )
    status_path = actual.get("egov-law-db/history/status.json")
    if status_path is None:
        return
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read e-Gov history status: {exc}")
        return
    if not isinstance(status, dict) or status.get("schema_version") != 1:
        errors.append("e-Gov history status schema_version must be 1")
        return
    expected_dates = list(config.egov_history_dates)
    expected_codes = set(config.egov_history_law_codes)
    if status.get("dates") != expected_dates:
        errors.append(
            f"e-Gov history dates mismatch: "
            f"expected={expected_dates} actual={status.get('dates')}"
        )
    actual_codes = set(status.get("law_codes") or [])
    if actual_codes != expected_codes:
        errors.append(
            f"e-Gov history law codes mismatch: "
            f"missing={sorted(expected_codes - actual_codes)} "
            f"extra={sorted(actual_codes - expected_codes)}"
        )
    snapshots = status.get("snapshots")
    if not isinstance(snapshots, list):
        errors.append("e-Gov history snapshots must be a list")
        return
    article_total = 0
    supplementary_total = 0
    snapshot_dates: list[str] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            errors.append("e-Gov history contains a non-object snapshot")
            continue
        as_of = snapshot.get("as_of")
        if not isinstance(as_of, str) or as_of not in config.egov_history_dates:
            errors.append(f"invalid e-Gov history snapshot date: {as_of!r}")
            continue
        if as_of in snapshot_dates:
            errors.append(f"duplicate e-Gov history snapshot date: {as_of}")
        snapshot_dates.append(as_of)
        date_status_path = f"egov-law-db/history/{as_of}/status.json"
        if date_status_path not in actual:
            errors.append(f"required e-Gov history date status is missing: {date_status_path}")
        else:
            try:
                date_status = json.loads(
                    actual[date_status_path].read_text(encoding="utf-8")
                )
                if date_status != snapshot:
                    errors.append(
                        f"e-Gov history date status does not match index: {as_of}"
                    )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"cannot read e-Gov history date status: {as_of}: {exc}")
        laws = snapshot.get("laws")
        if not isinstance(laws, list):
            errors.append(f"e-Gov history laws must be a list: {as_of}")
            continue
        snapshot_codes = {
            str(item.get("code"))
            for item in laws
            if isinstance(item, dict) and isinstance(item.get("code"), str)
        }
        if snapshot_codes != expected_codes:
            errors.append(f"e-Gov history snapshot law codes mismatch: {as_of}")
        snapshot_articles = 0
        snapshot_supplementary = 0
        for law in laws:
            if not isinstance(law, dict):
                errors.append(f"e-Gov history contains a non-object law: {as_of}")
                continue
            code = law.get("code")
            if not isinstance(code, str) or code not in expected_codes:
                continue
            if law.get("as_of") != as_of:
                errors.append(f"e-Gov history law date mismatch: {as_of}/{code}")
            for field in (
                "law_id",
                "law_num",
                "law_revision_id",
                "source_xml_sha256",
                "source_url",
            ):
                if not isinstance(law.get(field), str) or not law[field]:
                    errors.append(f"e-Gov history metadata is missing {field}: {as_of}/{code}")
            article_count = law.get("article_count")
            supplementary_count = law.get("supplementary_count")
            if not isinstance(article_count, int) or article_count < 1:
                errors.append(f"invalid e-Gov history article count: {as_of}/{code}")
                article_count = 0
            if not isinstance(supplementary_count, int) or supplementary_count < 0:
                errors.append(f"invalid e-Gov history supplementary count: {as_of}/{code}")
                supplementary_count = 0
            snapshot_articles += article_count
            snapshot_supplementary += supplementary_count
            paths = (
                f"egov-law-db/history/{as_of}/metadata/{code}.json",
                f"egov-law-db/history/{as_of}/text/{code}/index.html",
                f"egov-law-db/history/{as_of}/supplementary/{code}.txt",
            )
            for path in paths:
                if path not in actual:
                    errors.append(f"required e-Gov history law file is missing: {path}")
            prefix = f"egov-law-db/history/{as_of}/text/{code}/"
            text_count = sum(
                1
                for path in actual
                if path.startswith(prefix) and path.endswith(".txt")
            )
            if text_count != article_count:
                errors.append(
                    f"e-Gov history article files mismatch for {as_of}/{code}: "
                    f"expected={article_count} actual={text_count}"
                )
        if snapshot.get("article_count") != snapshot_articles:
            errors.append(f"e-Gov history snapshot article_count mismatch: {as_of}")
        if snapshot.get("supplementary_count") != snapshot_supplementary:
            errors.append(f"e-Gov history snapshot supplementary_count mismatch: {as_of}")
        article_total += snapshot_articles
        supplementary_total += snapshot_supplementary
    if snapshot_dates != expected_dates:
        errors.append(
            f"e-Gov history snapshot order/count mismatch: "
            f"expected={expected_dates} actual={snapshot_dates}"
        )
    revisions = status.get("revisions")
    revision_total = 0
    if not isinstance(revisions, list):
        errors.append("e-Gov history revisions must be a list")
        revisions = []
    revision_codes: set[str] = set()
    for revision in revisions:
        if not isinstance(revision, dict):
            errors.append("e-Gov history contains a non-object revision status")
            continue
        code = revision.get("code")
        if not isinstance(code, str) or code not in expected_codes:
            errors.append(f"invalid e-Gov revision code: {code!r}")
            continue
        if code in revision_codes:
            errors.append(f"duplicate e-Gov revision code: {code}")
        revision_codes.add(code)
        count = revision.get("revision_count")
        if not isinstance(count, int) or count < 1:
            errors.append(f"invalid e-Gov revision count: {code}")
            count = 0
        revision_total += count
        json_path = f"egov-law-db/history/revisions/{code}.json"
        text_path = f"egov-law-db/history/revisions/{code}.txt"
        for path in (json_path, text_path):
            if path not in actual:
                errors.append(f"required e-Gov revision file is missing: {path}")
        if json_path in actual and revision.get("json_sha256") != _sha256(actual[json_path]):
            errors.append(f"e-Gov revision JSON SHA-256 mismatch: {code}")
        if json_path in actual:
            try:
                payload = json.loads(actual[json_path].read_text(encoding="utf-8"))
                law_info = payload.get("law_info") if isinstance(payload, dict) else None
                entries = payload.get("revisions") if isinstance(payload, dict) else None
                if (
                    not isinstance(law_info, dict)
                    or law_info.get("law_id") != revision.get("law_id")
                    or not isinstance(entries, list)
                    or len(entries) != count
                ):
                    errors.append(f"e-Gov revision JSON content mismatch: {code}")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"cannot read e-Gov revision JSON: {code}: {exc}")
        if text_path in actual and revision.get("text_sha256") != _sha256(actual[text_path]):
            errors.append(f"e-Gov revision text SHA-256 mismatch: {code}")
    if revision_codes != expected_codes:
        errors.append("e-Gov history revision law codes are inconsistent")
    if status.get("date_count") != len(expected_dates):
        errors.append("e-Gov history date_count is inconsistent")
    if status.get("law_count_per_date") != len(expected_codes):
        errors.append("e-Gov history law_count_per_date is inconsistent")
    if article_total < config.egov_history_min_articles:
        errors.append(
            f"e-Gov history articles below minimum: "
            f"{article_total} < {config.egov_history_min_articles}"
        )
    if revision_total < config.egov_history_min_revisions:
        errors.append(
            f"e-Gov history revisions below minimum: "
            f"{revision_total} < {config.egov_history_min_revisions}"
        )
    if status.get("article_count") != article_total:
        errors.append("e-Gov history article_count is inconsistent")
    if status.get("supplementary_count") != supplementary_total:
        errors.append("e-Gov history supplementary_count is inconsistent")
    if status.get("revision_count") != revision_total:
        errors.append("e-Gov history revision_count is inconsistent")
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict):
        expected_metrics = {
            "egov_history_dates": len(config.egov_history_dates),
            "egov_history_laws": len(config.egov_history_law_codes),
            "egov_history_articles": article_total,
            "egov_history_supplementary_provisions": supplementary_total,
            "egov_history_revisions": revision_total,
        }
        for name, expected in expected_metrics.items():
            if metrics.get(name) != expected:
                errors.append(f"manifest {name} is inconsistent")


def _check_tax_question_tests(
    actual: dict[str, Path],
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or "tax_question_scenarios" not in metrics:
        return
    required = {
        "tax-question-tests/ai-rules.txt",
        "tax-question-tests/ai-packs/index.txt",
        "tax-question-tests/index.html",
        "tax-question-tests/prompts.txt",
        "tax-question-tests/report.json",
    }
    errors.extend(
        f"required tax-question test file is missing: {path}"
        for path in sorted(required - actual.keys())
    )
    report_path = actual.get("tax-question-tests/report.json")
    if report_path is None:
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read tax-question test report: {exc}")
        return
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        errors.append("tax-question test report schema_version must be 1")
        return
    if report.get("passed") is not True:
        errors.append("tax-question tests did not pass")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list):
        errors.append("tax-question test report scenarios must be a list")
        return
    source_count = 0
    answer_checklist_count = 0
    answer_review_item_count = 0
    ai_reference_pack_count = 0
    ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            errors.append("tax-question test report contains a non-object scenario")
            continue
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in ids:
            errors.append(f"invalid or duplicate tax-question scenario id: {scenario_id!r}")
        else:
            ids.add(scenario_id)
        if scenario.get("status") != "passed":
            errors.append(f"tax-question scenario did not pass: {scenario_id}")
        answer_checklist = scenario.get("answer_checklist", [])
        if not isinstance(answer_checklist, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in answer_checklist
        ):
            errors.append(f"invalid tax-question answer checklist: {scenario_id}")
        else:
            answer_checklist_count += len(answer_checklist)
        answer_review_guide = scenario.get("answer_review_guide", {})
        if not isinstance(answer_review_guide, dict):
            errors.append(f"invalid tax-question answer review guide: {scenario_id}")
        else:
            for section_name, section_items in answer_review_guide.items():
                if not isinstance(section_name, str) or not isinstance(
                    section_items, list
                ):
                    errors.append(
                        f"invalid tax-question answer review section: {scenario_id}"
                    )
                    continue
                if any(
                    not isinstance(item, str) or not item.strip()
                    for item in section_items
                ):
                    errors.append(
                        f"invalid tax-question answer review item: {scenario_id}"
                    )
                else:
                    answer_review_item_count += len(section_items)
        sources = scenario.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"tax-question scenario has no sources: {scenario_id}")
            continue
        source_count += len(sources)
        for source in sources:
            path = source.get("path") if isinstance(source, dict) else None
            if not isinstance(path, str) or path not in actual:
                errors.append(
                    f"tax-question scenario references a missing source: "
                    f"{scenario_id} -> {path!r}"
                )
    if report.get("scenario_count") != len(scenarios):
        errors.append("tax-question report scenario_count is inconsistent")
    if report.get("source_check_count") != source_count:
        errors.append("tax-question report source_check_count is inconsistent")
    if (
        "answer_checklist_count" in report
        and report.get("answer_checklist_count") != answer_checklist_count
    ):
        errors.append("tax-question report answer_checklist_count is inconsistent")
    if (
        "answer_review_item_count" in report
        and report.get("answer_review_item_count") != answer_review_item_count
    ):
        errors.append("tax-question report answer_review_item_count is inconsistent")
    ai_reference_packs = report.get("ai_reference_packs", [])
    if not isinstance(ai_reference_packs, list):
        errors.append("tax-question report ai_reference_packs must be a list")
    else:
        seen_pack_paths: set[str] = set()
        for pack in ai_reference_packs:
            if not isinstance(pack, dict):
                errors.append("tax-question report contains a non-object AI pack")
                continue
            path = pack.get("path")
            if (
                not isinstance(path, str)
                or not path.startswith("tax-question-tests/ai-packs/")
                or path in seen_pack_paths
                or path not in actual
            ):
                errors.append(f"invalid or missing tax-question AI pack: {path!r}")
            else:
                seen_pack_paths.add(path)
            scenario_ids = pack.get("scenario_ids")
            if not isinstance(scenario_ids, list) or any(
                not isinstance(scenario_id, str) or scenario_id not in ids
                for scenario_id in scenario_ids
            ):
                errors.append(f"invalid tax-question AI pack scenarios: {path!r}")
        ai_reference_pack_count = len(ai_reference_packs)
    if (
        "ai_reference_pack_count" in report
        and report.get("ai_reference_pack_count") != ai_reference_pack_count
    ):
        errors.append("tax-question report ai_reference_pack_count is inconsistent")
    if metrics.get("tax_question_scenarios") != len(scenarios):
        errors.append("manifest tax-question scenario count is inconsistent")
    if metrics.get("tax_question_source_checks") != source_count:
        errors.append("manifest tax-question source count is inconsistent")
    if (
        "tax_question_answer_check_items" in metrics
        and metrics.get("tax_question_answer_check_items") != answer_checklist_count
    ):
        errors.append("manifest tax-question answer checklist count is inconsistent")
    if (
        "tax_question_answer_review_items" in metrics
        and metrics.get("tax_question_answer_review_items") != answer_review_item_count
    ):
        errors.append("manifest tax-question answer review count is inconsistent")
    if (
        "tax_question_ai_reference_packs" in metrics
        and metrics.get("tax_question_ai_reference_packs") != ai_reference_pack_count
    ):
        errors.append("manifest tax-question AI reference pack count is inconsistent")


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
    _check_egov_history(actual, manifest, config, errors)
    _check_nta_sync(actual, manifest, config, errors)
    _check_tax_question_tests(actual, manifest, errors)
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
