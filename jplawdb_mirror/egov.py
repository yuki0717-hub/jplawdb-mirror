from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

import aiohttp

from .core import Config, MirrorError


JST = timezone(timedelta(hours=9))
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
ARTICLE_NUMBER_RE = re.compile(r"\d+(?:[-_]\d+)*(?::\d+(?:[-_]\d+)*)*")
STRUCTURAL_ELEMENT_RE = re.compile(r"Subitem(\d+)")


class EgovError(MirrorError):
    """The official e-Gov law synchronization could not be completed safely."""


@dataclass(frozen=True, slots=True)
class EgovArticle:
    key: str
    title: str
    text: str
    deleted: bool


@dataclass(frozen=True, slots=True)
class EgovLaw:
    code: str
    expected_title: str
    law_type: str
    law_id: str
    law_num: str
    law_revision_id: str
    law_title: str
    updated: str
    amendment_enforcement_date: str | None
    as_of: str
    xml: bytes
    articles: tuple[EgovArticle, ...]
    supplementary_count: int
    supplementary_text: str


def resolve_egov_as_of(value: str) -> str:
    if value == "current":
        return datetime.now(JST).date().isoformat()
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("egov.as_of must be 'current' or YYYY-MM-DD") from exc
    if parsed > datetime.now(JST).date():
        raise ValueError("egov.as_of cannot be in the future")
    return parsed.isoformat()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_descendant(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    for child in element.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _child_text(element: ElementTree.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return _normalized_text(child)
    return ""


def _normalized_text(element: ElementTree.Element) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _path_segment(element: ElementTree.Element) -> str | None:
    name = _local_name(element.tag)
    number = str(element.attrib.get("Num") or "").strip().replace("_", "-")
    if name == "Article":
        return f"a{number}" if number else "article"
    if name == "Paragraph":
        return f"p{number}" if number else "paragraph"
    if name == "Item":
        return f"i{number}" if number else "item"
    match = STRUCTURAL_ELEMENT_RE.fullmatch(name)
    if match:
        level = match.group(1)
        return f"s{level}-{number}" if number else f"subitem{level}"
    if name == "Class":
        return f"c{number}" if number else "class"
    return None


def _text_role(name: str) -> str | None:
    if name == "Sentence":
        return "sentence"
    if name.endswith("Caption"):
        return "caption"
    if name.endswith("Title"):
        return "title"
    if name.endswith("Label"):
        return "label"
    if name.endswith("Num"):
        return "number"
    return None


def _structured_element_text(
    element: ElementTree.Element,
    initial_path: tuple[str, ...] = (),
) -> str:
    """Render an XML subtree without discarding its hierarchy."""
    lines: list[str] = []

    def append(path: tuple[str, ...], role: str, text: str) -> None:
        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized:
            marker = "-".join(path) or "article"
            lines.append(f"[{marker}:{role}] {normalized}")

    def walk(node: ElementTree.Element, path: tuple[str, ...]) -> None:
        name = _local_name(node.tag)
        segment = _path_segment(node)
        current_path = path + (segment,) if segment else path
        role = _text_role(name)
        if role is not None:
            append(current_path, role, _normalized_text(node))
            return
        if node.text and node.text.strip():
            append(current_path, "text", node.text)
        for child in node:
            walk(child, current_path)
            if child.tail and child.tail.strip():
                append(current_path, "text", child.tail)

    walk(element, initial_path)
    if not lines:
        return _normalized_text(element)
    return "\n".join(lines)


def _structured_article_text(element: ElementTree.Element) -> str:
    return _structured_element_text(element)


def _supplementary_text(law_body: ElementTree.Element) -> tuple[int, str]:
    provisions = [
        element
        for element in law_body
        if _local_name(element.tag) == "SupplProvision"
    ]
    documents: list[str] = []
    for index, provision in enumerate(provisions, start=1):
        amendment_law_num = str(provision.attrib.get("AmendLawNum") or "").strip()
        extract = str(provision.attrib.get("Extract") or "").strip()
        label = _child_text(provision, "SupplProvisionLabel")
        header = (
            f"[supplementary:{index:03d}]\n"
            f"label: {label}\n"
            f"amendment_law_num: {amendment_law_num}\n"
            f"extract: {extract}\n"
        )
        body = _structured_element_text(provision, (f"suppl{index:03d}",))
        documents.append(f"{header}\n{body}".rstrip())
    return len(provisions), "\n\n".join(documents)


def parse_egov_xml(
    xml: bytes,
    *,
    code: str,
    expected_title: str,
    law_type: str,
    law_id: str,
    law_revision_id: str,
    law_num: str,
    updated: str,
    amendment_enforcement_date: str | None,
    as_of: str,
) -> EgovLaw:
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise EgovError(f"invalid e-Gov XML for {code}: {exc}") from exc
    if _local_name(root.tag) != "Law":
        raise EgovError(f"unexpected e-Gov XML root for {code}: {_local_name(root.tag)}")
    xml_law_type = str(root.attrib.get("LawType") or "")
    if xml_law_type != law_type:
        raise EgovError(
            f"e-Gov law type mismatch for {code}: expected={law_type!r} actual={xml_law_type!r}"
        )
    xml_law_num = _child_text(root, "LawNum")
    if xml_law_num != law_num:
        raise EgovError(
            f"e-Gov law number mismatch for {code}: expected={law_num!r} actual={xml_law_num!r}"
        )
    law_body = _first_descendant(root, "LawBody")
    if law_body is None:
        raise EgovError(f"e-Gov XML has no LawBody: {code}")
    title_element = _first_descendant(law_body, "LawTitle")
    actual_title = _normalized_text(title_element) if title_element is not None else ""
    if actual_title != expected_title:
        raise EgovError(
            f"e-Gov law title mismatch for {code}: expected={expected_title!r} actual={actual_title!r}"
        )
    main_provision = _first_descendant(law_body, "MainProvision")
    if main_provision is None:
        raise EgovError(f"e-Gov XML has no MainProvision: {code}")
    articles: list[EgovArticle] = []
    seen: set[str] = set()
    for element in main_provision.iter():
        if _local_name(element.tag) != "Article":
            continue
        raw_number = str(element.attrib.get("Num") or "").strip()
        if not ARTICLE_NUMBER_RE.fullmatch(raw_number):
            raise EgovError(f"unsupported e-Gov article number for {code}: {raw_number!r}")
        key = raw_number.replace(":", "-to-").replace("_", "-")
        if key in seen:
            raise EgovError(f"duplicate e-Gov article number for {code}: {key}")
        seen.add(key)
        article_title = _child_text(element, "ArticleTitle")
        body = _structured_article_text(element)
        deleted = str(element.attrib.get("Delete") or "").lower() == "true"
        if not body and not deleted:
            raise EgovError(f"empty e-Gov article for {code}: {key}")
        articles.append(
            EgovArticle(
                key=key,
                title=article_title,
                text=body or "（削除）",
                deleted=deleted,
            )
        )
    if not articles:
        raise EgovError(f"e-Gov law has no main articles: {code}")
    supplementary_count, supplementary_text = _supplementary_text(law_body)
    return EgovLaw(
        code=code,
        expected_title=expected_title,
        law_type=law_type,
        law_id=law_id,
        law_num=law_num,
        law_revision_id=law_revision_id,
        law_title=actual_title,
        updated=updated,
        amendment_enforcement_date=amendment_enforcement_date,
        as_of=as_of,
        xml=xml,
        articles=tuple(articles),
        supplementary_count=supplementary_count,
        supplementary_text=supplementary_text,
    )


def select_exact_law(
    payload: Any,
    *,
    code: str,
    title: str,
    law_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    laws = payload.get("laws") if isinstance(payload, dict) else None
    if not isinstance(laws, list):
        raise EgovError(f"e-Gov law search returned an unsupported schema: {code}")
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for candidate in laws:
        if not isinstance(candidate, dict):
            continue
        info = candidate.get("law_info")
        revision = candidate.get("revision_info")
        if not isinstance(info, dict) or not isinstance(revision, dict):
            continue
        if revision.get("law_title") == title and info.get("law_type") == law_type:
            matches.append((info, revision))
    if len(matches) != 1:
        raise EgovError(
            f"e-Gov exact law resolution failed for {code}: title={title!r} "
            f"type={law_type!r} matches={len(matches)}"
        )
    return matches[0]


class EgovClient:
    def __init__(self, config: Config, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.semaphore = asyncio.Semaphore(config.egov_concurrency)

    async def _request(self, url: str, *, params: dict[str, str] | None = None) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.semaphore:
                    timeout = aiohttp.ClientTimeout(total=self.config.timeout_sec)
                    async with self.session.get(
                        url,
                        params=params,
                        timeout=timeout,
                        headers={
                            "Accept": "application/json, application/xml;q=0.9, */*;q=0.8",
                            "User-Agent": "jplawdb-mirror/2 (+https://github.com/yuki0717-hub/jplawdb-mirror)",
                        },
                    ) as response:
                        if response.status == 200:
                            data = await response.read()
                            maximum = max(self.config.max_file_bytes * 2, 50_000_000)
                            if len(data) > maximum:
                                raise EgovError(
                                    f"e-Gov response exceeds maximum size: {url} ({len(data)} bytes)"
                                )
                            return data
                        if response.status not in RETRYABLE_STATUS:
                            raise EgovError(f"e-Gov HTTP {response.status}: {response.url}")
                        last_error = EgovError(f"e-Gov HTTP {response.status}: {response.url}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
            if attempt < self.config.max_retries:
                await asyncio.sleep(min(2**attempt, 8))
        raise EgovError(f"e-Gov request failed after retries: {url}: {last_error}")

    async def fetch_law(self, code: str, spec: dict[str, str], as_of: str) -> EgovLaw:
        search_url = f"{self.config.egov_api_base}/laws"
        search_bytes = await self._request(
            search_url,
            params={"law_title": spec["title"], "asof": as_of},
        )
        try:
            payload = json.loads(search_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EgovError(f"invalid e-Gov search JSON for {code}: {exc}") from exc
        info, revision = select_exact_law(
            payload,
            code=code,
            title=spec["title"],
            law_type=spec["type"],
        )
        law_id = str(info.get("law_id") or "")
        law_num = str(info.get("law_num") or "")
        revision_id = str(revision.get("law_revision_id") or "")
        if not law_id or not law_num or not revision_id:
            raise EgovError(f"e-Gov metadata is incomplete for {code}")
        xml_url = (
            f"{self.config.egov_api_base}/law_file/xml/"
            f"{quote(law_id, safe='')}"
        )
        xml = await self._request(xml_url, params={"asof": as_of})
        return parse_egov_xml(
            xml,
            code=code,
            expected_title=spec["title"],
            law_type=spec["type"],
            law_id=law_id,
            law_revision_id=revision_id,
            law_num=law_num,
            updated=str(revision.get("updated") or ""),
            amendment_enforcement_date=(
                str(revision["amendment_enforcement_date"])
                if revision.get("amendment_enforcement_date")
                else None
            ),
            as_of=as_of,
        )

    async def fetch_revisions(self, code: str, law_id: str) -> dict[str, Any]:
        url = f"{self.config.egov_api_base}/law_revisions/{quote(law_id, safe='')}"
        body = await self._request(url)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EgovError(f"invalid e-Gov revision JSON for {code}: {exc}") from exc
        law_info = payload.get("law_info") if isinstance(payload, dict) else None
        revisions = payload.get("revisions") if isinstance(payload, dict) else None
        if not isinstance(law_info, dict) or not isinstance(revisions, list):
            raise EgovError(f"unsupported e-Gov revision schema for {code}")
        if law_info.get("law_id") != law_id:
            raise EgovError(
                f"e-Gov revision law ID mismatch for {code}: "
                f"expected={law_id!r} actual={law_info.get('law_id')!r}"
            )
        if not revisions:
            raise EgovError(f"e-Gov revision history is empty for {code}")
        seen: set[str] = set()
        for revision in revisions:
            revision_id = revision.get("law_revision_id") if isinstance(revision, dict) else None
            if (
                not isinstance(revision_id, str)
                or not revision_id.startswith(f"{law_id}_")
                or revision_id in seen
            ):
                raise EgovError(
                    f"invalid or duplicate e-Gov revision ID for {code}: {revision_id!r}"
                )
            seen.add(revision_id)
        return payload


async def _fetch_all_laws(
    config: Config,
    session: aiohttp.ClientSession,
) -> tuple[str, list[EgovLaw]]:
    if not config.egov_laws:
        return resolve_egov_as_of(config.egov_as_of), []
    as_of = resolve_egov_as_of(config.egov_as_of)
    client = EgovClient(config, session)
    laws = await asyncio.gather(
        *(
            client.fetch_law(code, spec, as_of)
            for code, spec in sorted(config.egov_laws.items())
        )
    )
    if len(laws) != len(config.egov_laws):
        raise EgovError(
            f"e-Gov law count mismatch: expected={len(config.egov_laws)} actual={len(laws)}"
        )
    article_count = sum(len(law.articles) for law in laws)
    if article_count < config.egov_min_articles:
        raise EgovError(
            f"e-Gov article count below minimum: "
            f"actual={article_count} minimum={config.egov_min_articles}"
        )
    return as_of, laws


def _law_status(law: EgovLaw) -> dict[str, Any]:
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
        "xml_size": len(law.xml),
        "xml_sha256": hashlib.sha256(law.xml).hexdigest(),
        "source_url": (
            f"https://laws.e-gov.go.jp/law/{law.law_id}"
            f"?occasion_date={law.as_of.replace('-', '')}"
        ),
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_output(root: Path, as_of: str, laws: list[EgovLaw], mirror_base: str) -> None:
    base = root / "egov-law-db"
    generated_at = datetime.now(timezone.utc).isoformat()
    statuses: list[dict[str, Any]] = []
    root_links: list[str] = []
    for law in sorted(laws, key=lambda item: item.code):
        status = _law_status(law)
        statuses.append(status)
        xml_path = base / "xml" / f"{law.code}.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_bytes(law.xml)
        metadata_path = base / "metadata" / f"{law.code}.json"
        _write_text(
            metadata_path,
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        )
        supplementary_document = (
            "source: e-Gov法令検索\n"
            f"source_url: {status['source_url']}\n"
            f"law: {law.law_title} ({law.code})\n"
            f"law_num: {law.law_num}\n"
            f"law_id: {law.law_id}\n"
            f"law_revision_id: {law.law_revision_id}\n"
            f"as_of: {law.as_of}\n"
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
            text_path = base / "text" / law.code / f"{article.key}.txt"
            source_url = status["source_url"]
            document = (
                f"source: e-Gov法令検索\n"
                f"source_url: {source_url}\n"
                f"law: {law.law_title} ({law.code})\n"
                f"law_num: {law.law_num}\n"
                f"law_id: {law.law_id}\n"
                f"law_revision_id: {law.law_revision_id}\n"
                f"as_of: {law.as_of}\n"
                f"article: {article.key}\n"
                f"title: {article.title}\n"
                f"deleted: {'true' if article.deleted else 'false'}\n\n"
                f"{article.text}\n"
            )
            _write_text(text_path, document)
            article_links.append(
                f'<li><a href="{html.escape(article.key, quote=True)}.txt">'
                f"{html.escape(article.title or article.key)}</a></li>"
            )
        law_index = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(law.law_title)} - e-Gov同期</title></head><body>
<h1>{html.escape(law.law_title)}</h1>
<p>基準日: {html.escape(as_of)} / 法令ID: <code>{html.escape(law.law_id)}</code></p>
<p><a href="../../xml/{html.escape(law.code, quote=True)}.xml">公式XML</a> /
<a href="../../metadata/{html.escape(law.code, quote=True)}.json">同期メタデータ</a> /
<a href="../../supplementary/{html.escape(law.code, quote=True)}.txt">附則</a></p>
<ul>{''.join(article_links)}</ul>
</body></html>
"""
        _write_text(base / "text" / law.code / "index.html", law_index)
        root_links.append(
            f'<li><a href="text/{html.escape(law.code, quote=True)}/index.html">'
            f"{html.escape(law.law_title)}</a> "
            f"({len(law.articles):,}条文)</li>"
        )
    status_document = {
        "schema_version": 1,
        "generated_at": generated_at,
        "as_of": as_of,
        "api_base": "https://laws.e-gov.go.jp/api/2",
        "law_count": len(laws),
        "article_count": sum(len(law.articles) for law in laws),
        "supplementary_count": sum(law.supplementary_count for law in laws),
        "laws": statuses,
    }
    _write_text(
        base / "status.json",
        json.dumps(status_document, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text(
        base / "index.json",
        json.dumps(
            {
                "schema_version": 1,
                "as_of": as_of,
                "base_url": f"{mirror_base.rstrip('/')}/egov-law-db",
                "laws": statuses,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _write_text(
        base / "quickstart.txt",
        (
            "# e-Gov公式法令クイックスタート\n\n"
            f"基準日: {as_of}\n"
            "法令一覧: index.json\n"
            "同期状態: status.json\n"
            "条文: text/{law_code}/{article}.txt\n"
            "附則: supplementary/{law_code}.txt\n"
            "過去時点: history/index.html\n"
            "公式XML: xml/{law_code}.xml\n\n"
            "税務質問では、旧 ai-law-db よりこの egov-law-db の条文を優先してください。\n"
            "最終判断では各ファイルの source_url からe-Gov原文も確認してください。\n"
        ),
    )
    _write_text(
        base / "llms.txt",
        (
            "# e-Gov法令API Version 2 直接同期データ\n\n"
            f"- as_of: {as_of}\n"
            f"- laws: {len(laws)}\n"
            f"- main_articles: {sum(len(law.articles) for law in laws)}\n"
            f"- supplementary_provisions: {sum(law.supplementary_count for law in laws)}\n"
            "- canonical article: text/{law_code}/{article}.txt\n"
            "- supplementary provisions: supplementary/{law_code}.txt\n"
            "- historical snapshots: history/index.html\n"
            "- official source XML: xml/{law_code}.xml\n"
            "- metadata and hashes: status.json\n"
        ),
    )
    index_html = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>e-Gov公式法令同期</title></head><body>
<h1>e-Gov公式法令同期</h1>
<p>e-Gov法令API Version 2から直接取得した法令です。基準日: {html.escape(as_of)}</p>
<p><a href="quickstart.txt">AI向け利用案内</a> /
<a href="status.json">同期状態</a> / <a href="index.json">JSON索引</a></p>
<ul>{''.join(root_links)}</ul>
</body></html>
"""
    _write_text(base / "index.html", index_html)


def _metrics(laws: list[EgovLaw]) -> dict[str, int]:
    return {
        "egov_law_codes": len(laws),
        "egov_main_articles": sum(len(law.articles) for law in laws),
        "egov_supplementary_provisions": sum(
            law.supplementary_count for law in laws
        ),
        "egov_xml_bytes": sum(len(law.xml) for law in laws),
    }


async def inspect_egov_laws(
    config: Config,
    session: aiohttp.ClientSession,
) -> dict[str, int]:
    _, laws = await _fetch_all_laws(config, session)
    metrics = _metrics(laws)
    if laws:
        logging.info("Validated %s e-Gov laws", len(laws))
    return metrics


async def sync_egov_laws(
    config: Config,
    session: aiohttp.ClientSession,
    root: Path,
) -> dict[str, int]:
    as_of, laws = await _fetch_all_laws(config, session)
    if laws:
        await asyncio.to_thread(_write_output, root, as_of, laws, config.mirror_base)
        logging.info(
            "Generated e-Gov mirror for %s laws and %s main articles",
            len(laws),
            sum(len(law.articles) for law in laws),
        )
    return _metrics(laws)
