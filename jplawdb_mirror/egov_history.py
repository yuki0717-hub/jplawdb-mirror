from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from .core import Config
from .egov import EgovClient, EgovError, EgovLaw


@dataclass(frozen=True, slots=True)
class EgovHistory:
    snapshots: dict[str, tuple[EgovLaw, ...]]
    revisions: dict[str, dict[str, Any]]


def _source_url(law: EgovLaw) -> str:
    return (
        f"https://laws.e-gov.go.jp/law/{law.law_id}"
        f"?occasion_date={law.as_of.replace('-', '')}"
    )


def _history_law_status(law: EgovLaw) -> dict[str, Any]:
    return {
        "code": law.code,
        "title": law.law_title,
        "law_type": law.law_type,
        "law_id": law.law_id,
        "law_num": law.law_num,
        "law_revision_id": law.law_revision_id,
        "updated": law.updated,
        "amendment_enforcement_date": law.amendment_enforcement_date,
        "as_of": law.as_of,
        "article_count": len(law.articles),
        "supplementary_count": law.supplementary_count,
        "source_xml_size": len(law.xml),
        "source_xml_sha256": hashlib.sha256(law.xml).hexdigest(),
        "source_url": _source_url(law),
    }


async def _fetch_history(
    config: Config,
    session: aiohttp.ClientSession,
) -> EgovHistory:
    if not config.egov_history_dates:
        return EgovHistory(snapshots={}, revisions={})
    client = EgovClient(config, session)
    pairs = [
        (as_of, code, config.egov_laws[code])
        for as_of in config.egov_history_dates
        for code in config.egov_history_law_codes
    ]
    laws = await asyncio.gather(
        *(client.fetch_law(code, spec, as_of) for as_of, code, spec in pairs)
    )
    snapshots: dict[str, list[EgovLaw]] = {
        as_of: [] for as_of in config.egov_history_dates
    }
    for law in laws:
        snapshots[law.as_of].append(law)
    expected_codes = set(config.egov_history_law_codes)
    for as_of, snapshot_laws in snapshots.items():
        actual_codes = {law.code for law in snapshot_laws}
        if actual_codes != expected_codes:
            raise EgovError(
                f"e-Gov history law codes mismatch for {as_of}: "
                f"missing={sorted(expected_codes - actual_codes)} "
                f"extra={sorted(actual_codes - expected_codes)}"
            )
    law_ids: dict[str, str] = {}
    for law in laws:
        existing = law_ids.setdefault(law.code, law.law_id)
        if existing != law.law_id:
            raise EgovError(
                f"e-Gov history law ID changed for {law.code}: "
                f"{existing!r} != {law.law_id!r}"
            )
    revision_payloads = await asyncio.gather(
        *(
            client.fetch_revisions(code, law_ids[code])
            for code in config.egov_history_law_codes
        )
    )
    revisions = {
        code: payload
        for code, payload in zip(
            config.egov_history_law_codes,
            revision_payloads,
            strict=True,
        )
    }
    article_count = sum(len(law.articles) for law in laws)
    revision_count = sum(len(payload["revisions"]) for payload in revisions.values())
    if article_count < config.egov_history_min_articles:
        raise EgovError(
            f"e-Gov history article count below minimum: "
            f"actual={article_count} minimum={config.egov_history_min_articles}"
        )
    if revision_count < config.egov_history_min_revisions:
        raise EgovError(
            f"e-Gov history revision count below minimum: "
            f"actual={revision_count} minimum={config.egov_history_min_revisions}"
        )
    return EgovHistory(
        snapshots={
            as_of: tuple(sorted(snapshot_laws, key=lambda law: law.code))
            for as_of, snapshot_laws in snapshots.items()
        },
        revisions=revisions,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _revision_text(payload: dict[str, Any]) -> str:
    law_info = payload["law_info"]
    lines = [
        f"law_id: {law_info.get('law_id', '')}",
        f"law_num: {law_info.get('law_num', '')}",
        "",
        (
            "law_revision_id\tamendment_enforcement_date\t"
            "current_revision_status\tamendment_law_num\tamendment_law_title"
        ),
    ]
    for revision in payload["revisions"]:
        values = [
            revision.get("law_revision_id"),
            revision.get("amendment_enforcement_date"),
            revision.get("current_revision_status"),
            revision.get("amendment_law_num"),
            revision.get("amendment_law_title"),
        ]
        lines.append(
            "\t".join(
                str(value or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
                for value in values
            )
        )
    return "\n".join(lines) + "\n"


def _write_snapshot(
    history_base: Path,
    as_of: str,
    laws: tuple[EgovLaw, ...],
) -> dict[str, Any]:
    base = history_base / as_of
    law_links: list[str] = []
    statuses: list[dict[str, Any]] = []
    for law in laws:
        status = _history_law_status(law)
        statuses.append(status)
        _write_text(
            base / "metadata" / f"{law.code}.json",
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        )
        supplementary_document = (
            "source: e-Gov法令検索\n"
            f"source_url: {status['source_url']}\n"
            f"snapshot_kind: historical\n"
            f"law: {law.law_title} ({law.code})\n"
            f"law_num: {law.law_num}\n"
            f"law_id: {law.law_id}\n"
            f"law_revision_id: {law.law_revision_id}\n"
            f"as_of: {as_of}\n"
            f"supplementary_count: {law.supplementary_count}\n"
            "---\n"
            f"{law.supplementary_text or '附則なし'}\n"
        )
        _write_text(
            base / "supplementary" / f"{law.code}.txt",
            supplementary_document,
        )
        article_links: list[str] = []
        for article in law.articles:
            document = (
                "source: e-Gov法令検索\n"
                f"source_url: {status['source_url']}\n"
                "snapshot_kind: historical\n"
                f"law: {law.law_title} ({law.code})\n"
                f"law_num: {law.law_num}\n"
                f"law_id: {law.law_id}\n"
                f"law_revision_id: {law.law_revision_id}\n"
                f"as_of: {as_of}\n"
                f"article: {article.key}\n"
                f"title: {article.title}\n"
                f"deleted: {'true' if article.deleted else 'false'}\n\n"
                f"{article.text}\n"
            )
            _write_text(
                base / "text" / law.code / f"{article.key}.txt",
                document,
            )
            article_links.append(
                f'<li><a href="{html.escape(article.key, quote=True)}.txt">'
                f"{html.escape(article.title or article.key)}</a></li>"
            )
        law_page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(law.law_title)} - {html.escape(as_of)}</title></head><body>
<h1>{html.escape(law.law_title)}</h1>
<p>過去基準日: {html.escape(as_of)} /
法令履歴ID: <code>{html.escape(law.law_revision_id)}</code></p>
<p><a href="../../metadata/{html.escape(law.code, quote=True)}.json">メタデータ</a> /
<a href="../../supplementary/{html.escape(law.code, quote=True)}.txt">附則</a> /
<a href="{html.escape(status['source_url'], quote=True)}">e-Gov原文</a></p>
<ul>{''.join(article_links)}</ul>
</body></html>
"""
        _write_text(base / "text" / law.code / "index.html", law_page)
        law_links.append(
            f'<li><a href="text/{html.escape(law.code, quote=True)}/index.html">'
            f"{html.escape(law.law_title)}</a> ({len(law.articles):,}条文)</li>"
        )
    status_document = {
        "schema_version": 1,
        "as_of": as_of,
        "law_count": len(laws),
        "article_count": sum(len(law.articles) for law in laws),
        "supplementary_count": sum(law.supplementary_count for law in laws),
        "laws": statuses,
    }
    _write_text(
        base / "status.json",
        json.dumps(status_document, ensure_ascii=False, indent=2) + "\n",
    )
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>e-Gov過去法令 {html.escape(as_of)}</title></head><body>
<h1>e-Gov過去法令</h1>
<p>基準日: {html.escape(as_of)} / <a href="status.json">同期状態</a></p>
<ul>{''.join(law_links)}</ul>
</body></html>
"""
    _write_text(base / "index.html", page)
    return status_document


def _write_output(root: Path, history: EgovHistory, mirror_base: str) -> None:
    base = root / "egov-law-db" / "history"
    generated_at = datetime.now(timezone.utc).isoformat()
    snapshot_statuses = [
        _write_snapshot(base, as_of, history.snapshots[as_of])
        for as_of in history.snapshots
    ]
    revision_statuses: list[dict[str, Any]] = []
    revision_links: list[str] = []
    for code, payload in sorted(history.revisions.items()):
        json_document = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        text_document = _revision_text(payload)
        _write_text(base / "revisions" / f"{code}.json", json_document)
        _write_text(base / "revisions" / f"{code}.txt", text_document)
        revision_statuses.append(
            {
                "code": code,
                "law_id": payload["law_info"].get("law_id"),
                "law_num": payload["law_info"].get("law_num"),
                "revision_count": len(payload["revisions"]),
                "json_sha256": hashlib.sha256(json_document.encode("utf-8")).hexdigest(),
                "text_sha256": hashlib.sha256(text_document.encode("utf-8")).hexdigest(),
            }
        )
        revision_links.append(
            f'<li>{html.escape(code)}: '
            f'<a href="revisions/{html.escape(code, quote=True)}.json">JSON</a> / '
            f'<a href="revisions/{html.escape(code, quote=True)}.txt">text</a></li>'
        )
    status = {
        "schema_version": 1,
        "generated_at": generated_at,
        "api_base": "https://laws.e-gov.go.jp/api/2",
        "dates": list(history.snapshots),
        "law_codes": sorted(history.revisions),
        "date_count": len(history.snapshots),
        "law_count_per_date": len(history.revisions),
        "article_count": sum(item["article_count"] for item in snapshot_statuses),
        "supplementary_count": sum(
            item["supplementary_count"] for item in snapshot_statuses
        ),
        "revision_count": sum(
            item["revision_count"] for item in revision_statuses
        ),
        "snapshots": snapshot_statuses,
        "revisions": revision_statuses,
    }
    _write_text(
        base / "status.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text(
        base / "index.json",
        json.dumps(
            {
                **status,
                "base_url": f"{mirror_base.rstrip('/')}/egov-law-db/history",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _write_text(
        base / "quickstart.txt",
        (
            "# e-Gov過去法令クイックスタート\n\n"
            "過去条文: {YYYY-MM-DD}/text/{law_code}/{article}.txt\n"
            "過去附則: {YYYY-MM-DD}/supplementary/{law_code}.txt\n"
            "改正履歴: revisions/{law_code}.json\n"
            "全体状態: status.json\n\n"
            "質問の取引日・事業年度に一致する基準日を選び、"
            "現在法令と混同しないでください。\n"
        ),
    )
    date_links = "".join(
        f'<li><a href="{html.escape(as_of, quote=True)}/index.html">'
        f"{html.escape(as_of)}</a></li>"
        for as_of in history.snapshots
    )
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>e-Gov過去法令・改正履歴</title></head><body>
<h1>e-Gov過去法令・改正履歴</h1>
<p>平成29年4月1日以降に対応するe-Gov法令API Version 2から取得しています。</p>
<p><a href="quickstart.txt">AI向け案内</a> /
<a href="status.json">同期状態</a> / <a href="index.json">JSON索引</a></p>
<h2>過去基準日</h2><ul>{date_links}</ul>
<h2>改正履歴</h2><ul>{''.join(revision_links)}</ul>
</body></html>
"""
    _write_text(base / "index.html", page)


def _metrics(history: EgovHistory) -> dict[str, int]:
    laws = [law for snapshot in history.snapshots.values() for law in snapshot]
    return {
        "egov_history_dates": len(history.snapshots),
        "egov_history_laws": len(history.revisions),
        "egov_history_articles": sum(len(law.articles) for law in laws),
        "egov_history_supplementary_provisions": sum(
            law.supplementary_count for law in laws
        ),
        "egov_history_revisions": sum(
            len(payload["revisions"]) for payload in history.revisions.values()
        ),
        "egov_history_xml_bytes_fetched": sum(len(law.xml) for law in laws),
    }


async def inspect_egov_history(
    config: Config,
    session: aiohttp.ClientSession,
) -> dict[str, int]:
    history = await _fetch_history(config, session)
    metrics = _metrics(history)
    if history.snapshots:
        logging.info(
            "Validated e-Gov history for %s dates and %s laws",
            len(history.snapshots),
            len(history.revisions),
        )
    return metrics


async def sync_egov_history(
    config: Config,
    session: aiohttp.ClientSession,
    root: Path,
) -> dict[str, int]:
    history = await _fetch_history(config, session)
    if history.snapshots:
        await asyncio.to_thread(_write_output, root, history, config.mirror_base)
        logging.info(
            "Generated e-Gov history for %s dates and %s laws",
            len(history.snapshots),
            len(history.revisions),
        )
    return _metrics(history)
