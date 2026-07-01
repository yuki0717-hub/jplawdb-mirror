from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp

from .core import Config, MirrorError, NtaSourceSpec


JST = timezone(timedelta(hours=9))
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
CHARSET_RE = re.compile(br"""charset\s*=\s*["']?\s*([a-zA-Z0-9._-]+)""", re.I)
LEGAL_DATE_RE = re.compile(
    r"[\[［]令和(?P<year>\d+|元)年(?P<month>\d+)月(?P<day>\d+)日現在法令等[\]］]"
)
DASH_TRANSLATION = str.maketrans(
    {
        "\N{HYPHEN}": "-",
        "\N{NON-BREAKING HYPHEN}": "-",
        "\N{FIGURE DASH}": "-",
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{HORIZONTAL BAR}": "-",
        "\N{MINUS SIGN}": "-",
        "\N{FULLWIDTH HYPHEN-MINUS}": "-",
    }
)


class NtaError(MirrorError):
    """Official National Tax Agency synchronization failed."""


@dataclass(frozen=True, slots=True)
class NtaDocument:
    code: str
    expected_title: str
    source_url: str
    final_url: str
    title: str
    fetched_at: str
    last_modified: str | None
    etag: str | None
    declared_charset: str
    decoded_charset: str
    legal_as_of: str | None
    legal_age_days: int | None
    source_bytes: bytes
    decoded_html: str
    text: str


@dataclass(frozen=True, slots=True)
class _NtaResponse:
    body: bytes
    final_url: str
    content_type: str
    last_modified: str | None
    etag: str | None


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
SUPPRESSED_TAGS = {"script", "style", "noscript", "svg", "template"}


class _NtaHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.content_parts: list[str] = []
        self.title_depth = 0
        self.content_depth = 0
        self.suppressed_depth = 0

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {name.lower(): value or "" for name, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = self._attrs(attrs)
        if tag == "title":
            self.title_depth += 1
        if self.content_depth == 0 and attributes.get("id") == "contents":
            self.content_depth = 1
            self.content_parts.append("\n")
        elif self.content_depth and tag not in VOID_TAGS:
            self.content_depth += 1
        if self.content_depth:
            if self.suppressed_depth:
                if tag not in VOID_TAGS:
                    self.suppressed_depth += 1
            elif tag in SUPPRESSED_TAGS:
                self.suppressed_depth = 1
            elif tag in BLOCK_TAGS or tag == "br":
                self.content_parts.append("\n")
            if tag == "img" and attributes.get("alt"):
                self.content_parts.append(attributes["alt"])

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.content_depth or self.suppressed_depth:
            return
        attributes = self._attrs(attrs)
        if tag.lower() in BLOCK_TAGS or tag.lower() == "br":
            self.content_parts.append("\n")
        if tag.lower() == "img" and attributes.get("alt"):
            self.content_parts.append(attributes["alt"])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if not self.content_depth:
            return
        if self.suppressed_depth:
            self.suppressed_depth -= 1
        elif tag in BLOCK_TAGS:
            self.content_parts.append("\n")
        if tag not in VOID_TAGS:
            self.content_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_parts.append(data)
        if self.content_depth and not self.suppressed_depth:
            self.content_parts.append(data)


def _match_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(DASH_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def decode_nta_html(data: bytes) -> tuple[str, str, str]:
    if not data:
        raise NtaError("empty NTA HTML response")
    match = CHARSET_RE.search(data[:8192])
    declared = match.group(1).decode("ascii").lower() if match else ""
    aliases = {
        "shift_jis": "cp932",
        "shift-jis": "cp932",
        "sjis": "cp932",
        "windows-31j": "cp932",
        "x-sjis": "cp932",
        "utf8": "utf-8",
    }
    candidates = [aliases.get(declared, declared)] if declared else []
    candidates.extend(encoding for encoding in ("utf-8", "cp932") if encoding not in candidates)
    last_error: UnicodeDecodeError | None = None
    for encoding in candidates:
        if not encoding:
            continue
        try:
            return data.decode(encoding), declared or encoding, encoding
        except (LookupError, UnicodeDecodeError) as exc:
            if isinstance(exc, UnicodeDecodeError):
                last_error = exc
    raise NtaError(f"cannot decode NTA HTML: declared={declared!r}: {last_error}")


def extract_nta_text(decoded_html: str) -> tuple[str, str]:
    parser = _NtaHtmlParser()
    try:
        parser.feed(decoded_html)
        parser.close()
    except Exception as exc:
        raise NtaError(f"cannot parse NTA HTML: {exc}") from exc
    title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
    lines = []
    for raw_line in "".join(parser.content_parts).splitlines():
        line = re.sub(r"[\t\r\f\v ]+", " ", raw_line).strip()
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    text = "\n".join(lines)
    if not title or not text:
        raise NtaError("NTA HTML has no title or #contents text")
    return title, text


def _legal_date(text: str) -> date | None:
    match = LEGAL_DATE_RE.search(unicodedata.normalize("NFKC", text))
    if match is None:
        return None
    reiwa_year = 1 if match.group("year") == "元" else int(match.group("year"))
    year = reiwa_year + 2018
    try:
        return date(year, int(match.group("month")), int(match.group("day")))
    except ValueError as exc:
        raise NtaError(f"invalid NTA legal date: {match.group(0)!r}") from exc


def parse_nta_document(
    response: _NtaResponse,
    *,
    code: str,
    spec: NtaSourceSpec,
    maximum_legal_age_days: int,
    today: date | None = None,
) -> NtaDocument:
    decoded_html, declared_charset, decoded_charset = decode_nta_html(response.body)
    title, text = extract_nta_text(decoded_html)
    normalized_title = _match_text(title)
    if _match_text(spec.title) not in normalized_title:
        raise NtaError(
            f"NTA title mismatch for {code}: expected={spec.title!r} actual={title!r}"
        )
    normalized_text = _match_text(text)
    missing_terms = [
        term for term in spec.required_terms if _match_text(term) not in normalized_text
    ]
    if missing_terms:
        raise NtaError(f"NTA required terms are missing for {code}: {missing_terms}")
    if len(text) < spec.minimum_text_chars:
        raise NtaError(
            f"NTA extracted text is too short for {code}: "
            f"actual={len(text)} minimum={spec.minimum_text_chars}"
        )
    legal_date = _legal_date(text)
    if spec.require_legal_date and legal_date is None:
        raise NtaError(f"NTA legal date is missing for {code}")
    current = today or datetime.now(JST).date()
    age_days = (current - legal_date).days if legal_date else None
    if age_days is not None and (age_days < 0 or age_days > maximum_legal_age_days):
        raise NtaError(
            f"NTA legal date is stale for {code}: "
            f"current={current} legal_as_of={legal_date} age_days={age_days} "
            f"maximum={maximum_legal_age_days}"
        )
    return NtaDocument(
        code=code,
        expected_title=spec.title,
        source_url=spec.url,
        final_url=response.final_url,
        title=title,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        last_modified=response.last_modified,
        etag=response.etag,
        declared_charset=declared_charset,
        decoded_charset=decoded_charset,
        legal_as_of=legal_date.isoformat() if legal_date else None,
        legal_age_days=age_days,
        source_bytes=response.body,
        decoded_html=decoded_html,
        text=text,
    )


class NtaClient:
    def __init__(self, config: Config, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.semaphore = asyncio.Semaphore(config.nta_concurrency)

    async def _request(self, url: str) -> _NtaResponse:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.semaphore:
                    timeout = aiohttp.ClientTimeout(total=self.config.timeout_sec)
                    async with self.session.get(
                        url,
                        timeout=timeout,
                        headers={
                            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                            "User-Agent": (
                                "jplawdb-mirror/2 "
                                "(+https://github.com/yuki0717-hub/jplawdb-mirror)"
                            ),
                        },
                    ) as response:
                        if response.status == 200:
                            final_url = str(response.url)
                            parsed = urlsplit(final_url)
                            if parsed.scheme != "https" or parsed.hostname != "www.nta.go.jp":
                                raise NtaError(
                                    f"NTA redirect left the official host: {url} -> {final_url}"
                                )
                            content_type = response.headers.get("Content-Type", "")
                            if "text/html" not in content_type.lower():
                                raise NtaError(
                                    f"NTA response is not HTML: {final_url}: {content_type!r}"
                                )
                            body = await response.read()
                            if len(body) > 5_000_000:
                                raise NtaError(
                                    f"NTA response exceeds maximum size: "
                                    f"{final_url} ({len(body)} bytes)"
                                )
                            return _NtaResponse(
                                body=body,
                                final_url=final_url,
                                content_type=content_type,
                                last_modified=response.headers.get("Last-Modified"),
                                etag=response.headers.get("ETag"),
                            )
                        if response.status not in RETRYABLE_STATUS:
                            raise NtaError(f"NTA HTTP {response.status}: {response.url}")
                        last_error = NtaError(f"NTA HTTP {response.status}: {response.url}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
            if attempt < self.config.max_retries:
                await asyncio.sleep(min(2**attempt, 8))
        raise NtaError(f"NTA request failed after retries: {url}: {last_error}")

    async def fetch(self, code: str, spec: NtaSourceSpec) -> NtaDocument:
        response = await self._request(spec.url)
        return parse_nta_document(
            response,
            code=code,
            spec=spec,
            maximum_legal_age_days=self.config.nta_max_legal_age_days,
        )


async def _fetch_all(
    config: Config,
    session: aiohttp.ClientSession,
) -> list[NtaDocument]:
    if not config.nta_sources:
        return []
    client = NtaClient(config, session)
    documents = await asyncio.gather(
        *(client.fetch(code, spec) for code, spec in sorted(config.nta_sources.items()))
    )
    if len(documents) != len(config.nta_sources):
        raise NtaError(
            f"NTA document count mismatch: "
            f"expected={len(config.nta_sources)} actual={len(documents)}"
        )
    return documents


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_output(
    root: Path,
    documents: list[NtaDocument],
    mirror_base: str,
    maximum_legal_age_days: int,
) -> None:
    base = root / "nta-official-db"
    generated_at = datetime.now(timezone.utc).isoformat()
    statuses: list[dict[str, Any]] = []
    links: list[str] = []
    for document in sorted(documents, key=lambda item: item.code):
        source_sha256 = hashlib.sha256(document.source_bytes).hexdigest()
        text_document = (
            "source: 国税庁\n"
            f"source_url: {document.source_url}\n"
            f"final_url: {document.final_url}\n"
            f"code: {document.code}\n"
            f"title: {document.title}\n"
            f"fetched_at: {document.fetched_at}\n"
            f"last_modified: {document.last_modified or ''}\n"
            f"legal_as_of: {document.legal_as_of or ''}\n"
            f"declared_charset: {document.declared_charset}\n"
            f"source_sha256: {source_sha256}\n"
            "---\n"
            f"{document.text}\n"
        )
        text_path = base / "text" / f"{document.code}.txt"
        raw_path = base / "raw" / f"{document.code}.html.txt"
        _write_text(text_path, text_document)
        _write_text(raw_path, document.decoded_html.rstrip() + "\n")
        status = {
            "code": document.code,
            "title": document.title,
            "expected_title": document.expected_title,
            "source_url": document.source_url,
            "final_url": document.final_url,
            "fetched_at": document.fetched_at,
            "last_modified": document.last_modified,
            "etag": document.etag,
            "declared_charset": document.declared_charset,
            "decoded_charset": document.decoded_charset,
            "legal_as_of": document.legal_as_of,
            "legal_age_days": document.legal_age_days,
            "source_size": len(document.source_bytes),
            "source_sha256": source_sha256,
            "text_path": f"text/{document.code}.txt",
            "text_sha256": hashlib.sha256(text_document.encode("utf-8")).hexdigest(),
            "raw_path": f"raw/{document.code}.html.txt",
            "raw_sha256": hashlib.sha256(
                (document.decoded_html.rstrip() + "\n").encode("utf-8")
            ).hexdigest(),
        }
        statuses.append(status)
        _write_text(
            base / "metadata" / f"{document.code}.json",
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        )
        links.append(
            f'<li><a href="text/{quote(document.code, safe="")}.txt">'
            f"{html.escape(document.title)}</a> "
            f'(<a href="{html.escape(document.source_url, quote=True)}">国税庁原文</a>)</li>'
        )
    status_document = {
        "schema_version": 1,
        "generated_at": generated_at,
        "fetched_on": datetime.now(JST).date().isoformat(),
        "source_count": len(statuses),
        "maximum_legal_age_days": maximum_legal_age_days,
        "sources": statuses,
    }
    _write_text(
        base / "status.json",
        json.dumps(status_document, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text(
        base / "index.json",
        json.dumps(
            {
                **status_document,
                "base_url": f"{mirror_base.rstrip('/')}/nta-official-db",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _write_text(
        base / "quickstart.txt",
        (
            "# 国税庁公式資料 直接同期DB\n\n"
            "国税庁公式サイトからビルド時に直接取得した資料です。\n"
            "本文: text/{code}.txt\n"
            "取得状態・法令等現在日・SHA-256: status.json\n"
            "文字コード変換後のHTML原文: raw/{code}.html.txt\n\n"
            "税務質問では同じ資料の旧スナップショットより、このDBを優先してください。\n"
            "最終判断では各本文の source_url も確認してください。\n"
        ),
    )
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>国税庁公式資料 直接同期DB</title></head><body>
<h1>国税庁公式資料 直接同期DB</h1>
<p>国税庁公式サイトから直接取得し、文字コード・タイトル・必須語・鮮度を検証した資料です。</p>
<p>取得日: {html.escape(status_document["fetched_on"])} /
<a href="status.json">同期状態</a> / <a href="quickstart.txt">AI向け案内</a></p>
<ul>{''.join(links)}</ul>
</body></html>
"""
    _write_text(base / "index.html", page)


def _metrics(documents: list[NtaDocument]) -> dict[str, int]:
    return {
        "nta_official_documents": len(documents),
        "nta_official_source_bytes": sum(len(document.source_bytes) for document in documents),
        "nta_official_legal_dates": sum(
            1 for document in documents if document.legal_as_of is not None
        ),
    }


async def inspect_nta_sources(
    config: Config,
    session: aiohttp.ClientSession,
) -> dict[str, int]:
    documents = await _fetch_all(config, session)
    metrics = _metrics(documents)
    if documents:
        logging.info("Validated %s official NTA documents", len(documents))
    return metrics


async def sync_nta_sources(
    config: Config,
    session: aiohttp.ClientSession,
    root: Path,
) -> dict[str, int]:
    documents = await _fetch_all(config, session)
    if documents:
        await asyncio.to_thread(
            _write_output,
            root,
            documents,
            config.mirror_base,
            config.nta_max_legal_age_days,
        )
        logging.info("Generated official NTA mirror for %s documents", len(documents))
    return _metrics(documents)
