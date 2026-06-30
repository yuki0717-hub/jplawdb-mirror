from __future__ import annotations

import asyncio
import csv
import hashlib
import html
import io
import json
import logging
import posixpath
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urljoin, urlsplit

import aiohttp
import yaml

TEXT_EXTENSIONS = {".txt", ".json", ".jsonl", ".html", ".htm", ".tsv", ".csv", ".xml"}
ARTICLE_LINK_RE = re.compile(
    r"href\s*=\s*['\"](?:[^'\"]*/)?(\d+(?:[-:]\d+)*)\.html(?:#[^'\"]*)?['\"]",
    re.IGNORECASE,
)
HTML_REFERENCE_RE = re.compile(r"(?:href|src)\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
BEPPYO_ID_RE = re.compile(r"\|\s*(B\d+(?:-\d+)*)\s*\|")
LOCAL_ASSET_PREFIXES = ("packs/", "text/", "enhanced/", "data/", "core/")
URL_PATH_SAFE = "/@:+-._~!$&'()*,;="


class MirrorError(RuntimeError):
    """Base error for a build that must not be published."""


class FetchError(MirrorError):
    """A required source file could not be downloaded."""


class DiscoveryError(MirrorError):
    """Source metadata was missing or had an unsupported schema."""


@dataclass(frozen=True, slots=True)
class Config:
    source_base: str
    mirror_base: str
    output_dir: Path = Path("output")
    concurrency: int = 16
    delay_sec: float = 0.05
    max_retries: int = 4
    timeout_sec: int = 60
    max_file_bytes: int = 25_000_000
    minimum_counts: dict[str, int] = field(default_factory=dict)
    egov_api_base: str = ""
    egov_as_of: str = "current"
    egov_concurrency: int = 4
    egov_min_articles: int = 1
    egov_laws: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        if not path.exists():
            raise ValueError(f"config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("config must be a YAML mapping")
        source_base = str(raw.get("source_base", "")).rstrip("/")
        mirror_base = str(raw.get("mirror_base") or raw.get("my_base") or "").rstrip("/")
        if not source_base or not mirror_base:
            raise ValueError("source_base and mirror_base are required")
        for label, value in (("source_base", source_base), ("mirror_base", mirror_base)):
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{label} must be an absolute HTTP(S) URL")
        minimum_counts = raw.get("minimum_counts") or {}
        if not isinstance(minimum_counts, dict):
            raise ValueError("minimum_counts must be a mapping")
        egov = raw.get("egov") or {}
        if not isinstance(egov, dict):
            raise ValueError("egov must be a mapping")
        egov_api_base = str(egov.get("api_base") or "").rstrip("/")
        egov_as_of = str(egov.get("as_of") or "current")
        egov_concurrency = int(egov.get("concurrency", 4))
        egov_min_articles = int(egov.get("minimum_articles", 1))
        egov_laws_raw = egov.get("laws") or {}
        if not isinstance(egov_laws_raw, dict):
            raise ValueError("egov.laws must be a mapping")
        egov_laws: dict[str, dict[str, str]] = {}
        allowed_law_types = {
            "Act",
            "CabinetOrder",
            "ImperialOrder",
            "MinisterialOrdinance",
            "Rule",
            "Misc",
        }
        for code, law in egov_laws_raw.items():
            if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]+", code):
                raise ValueError(f"invalid egov law code: {code!r}")
            if not isinstance(law, dict):
                raise ValueError(f"egov law must be a mapping: {code}")
            title = str(law.get("title") or "").strip()
            law_type = str(law.get("type") or "").strip()
            if not title or law_type not in allowed_law_types:
                raise ValueError(f"invalid egov law specification: {code}")
            egov_laws[code] = {"title": title, "type": law_type}
        if egov_laws:
            parsed_egov = urlsplit(egov_api_base)
            if parsed_egov.scheme != "https" or not parsed_egov.netloc:
                raise ValueError("egov.api_base must be an absolute HTTPS URL")
            if egov_concurrency < 1 or egov_min_articles < 1:
                raise ValueError("invalid egov concurrency or minimum_articles")
        config = cls(
            source_base=source_base,
            mirror_base=mirror_base,
            output_dir=Path(str(raw.get("output_dir", "output"))),
            concurrency=int(raw.get("concurrency", 16)),
            delay_sec=float(raw.get("delay_sec", 0.05)),
            max_retries=int(raw.get("max_retries", 4)),
            timeout_sec=int(raw.get("timeout_sec", 60)),
            max_file_bytes=int(raw.get("max_file_bytes", 25_000_000)),
            minimum_counts={str(k): int(v) for k, v in minimum_counts.items()},
            egov_api_base=egov_api_base,
            egov_as_of=egov_as_of,
            egov_concurrency=egov_concurrency,
            egov_min_articles=egov_min_articles,
            egov_laws=egov_laws,
        )
        if config.concurrency < 1 or config.max_retries < 0 or config.timeout_sec < 1:
            raise ValueError("invalid concurrency, retry, or timeout setting")
        return config


@dataclass(frozen=True, slots=True)
class Target:
    path: str
    dataset: str


@dataclass(slots=True)
class DiscoveryPlan:
    targets: dict[str, Target] = field(default_factory=dict)
    metrics: Counter[str] = field(default_factory=Counter)

    def add(self, path: str, dataset: str) -> str:
        clean = safe_relative_path(path)
        existing = self.targets.get(clean)
        if existing is None:
            self.targets[clean] = Target(clean, dataset)
        return clean


@dataclass(frozen=True, slots=True)
class BuildResult:
    output_dir: Path
    file_count: int
    total_bytes: int
    metrics: dict[str, int]


class RequestScheduler:
    def __init__(self, delay_sec: float) -> None:
        self.delay_sec = max(0.0, delay_sec)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def wait(self) -> None:
        if self.delay_sec == 0:
            return
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait_for = self._next_allowed - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = asyncio.get_running_loop().time()
            self._next_allowed = now + self.delay_sec


class Fetcher:
    def __init__(self, session: aiohttp.ClientSession, config: Config) -> None:
        self.session = session
        self.config = config
        self.cache: dict[str, bytes] = {}
        self.scheduler = RequestScheduler(config.delay_sec)
        self.semaphore = asyncio.Semaphore(config.concurrency)
        self.statuses: dict[str, tuple[int, int]] = {}

    async def fetch_path(
        self,
        path: str,
        *,
        optional: bool = False,
        store_cache: bool = True,
    ) -> bytes | None:
        clean = safe_relative_path(path)
        if clean in self.cache:
            return self.cache[clean]
        url = source_url(self.config.source_base, clean)
        last_error = "unknown error"
        for attempt in range(self.config.max_retries + 1):
            try:
                await self.scheduler.wait()
                async with self.semaphore:
                    async with self.session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_sec),
                    ) as response:
                        status = response.status
                        if status == 404 and optional:
                            self.statuses[clean] = (status, 0)
                            return None
                        if 200 <= status < 300:
                            final_path = source_path_from_url(str(response.url), self.config.source_base)
                            if final_path is None:
                                raise FetchError(f"redirected outside source: {url} -> {response.url}")
                            body = await response.read()
                            if len(body) > self.config.max_file_bytes:
                                raise FetchError(
                                    f"source file exceeds max_file_bytes: {clean} ({len(body)} bytes)"
                                )
                            self.statuses[clean] = (status, len(body))
                            if store_cache:
                                self.cache[clean] = body
                            return body
                        retryable = status in {408, 425, 429} or status >= 500
                        last_error = f"HTTP {status}"
                        if not retryable or attempt == self.config.max_retries:
                            break
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            await asyncio.sleep(min(float(retry_after), 60.0))
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = str(exc)
                if attempt == self.config.max_retries:
                    break
            if attempt < self.config.max_retries:
                await asyncio.sleep(min(30.0, (2**attempt) * max(0.1, self.config.delay_sec)))
        if optional:
            logging.warning("optional source unavailable: %s (%s)", clean, last_error)
            return None
        raise FetchError(f"required source unavailable: {clean} ({last_error})")


class Discovery:
    def __init__(self, config: Config, fetcher: Fetcher) -> None:
        self.config = config
        self.fetcher = fetcher
        self.plan = DiscoveryPlan()

    async def required_bytes(self, path: str, dataset: str) -> bytes:
        clean = self.plan.add(path, dataset)
        body = await self.fetcher.fetch_path(clean)
        if body is None:
            raise FetchError(f"required source unavailable: {clean}")
        return body

    async def required_text(self, path: str, dataset: str) -> str:
        body = await self.required_bytes(path, dataset)
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DiscoveryError(f"source is not UTF-8: {path}") from exc

    async def required_json(self, path: str, dataset: str) -> Any:
        text = await self.required_text(path, dataset)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise DiscoveryError(f"invalid JSON: {path}: {exc}") from exc

    async def optional_bytes(self, path: str, dataset: str) -> bytes | None:
        clean = safe_relative_path(path)
        body = await self.fetcher.fetch_path(clean, optional=True)
        if body is not None:
            self.plan.add(clean, dataset)
        return body

    async def optional_json(self, path: str, dataset: str) -> Any | None:
        body = await self.optional_bytes(path, dataset)
        if body is None:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"invalid optional JSON: {path}: {exc}") from exc

    def add_reference(self, root: str, reference: str, dataset: str) -> str | None:
        path = resolve_dataset_reference(root, reference, self.config.source_base)
        if path is not None:
            self.plan.add(path, dataset)
        return path

    def add_html_references(self, page_path: str, page_html: str, dataset: str) -> None:
        page_url = source_url(self.config.source_base, page_path)
        for reference in HTML_REFERENCE_RE.findall(page_html):
            reference = html.unescape(reference.strip())
            if not reference or reference.endswith("/"):
                continue
            local_path = source_path_from_url(urljoin(page_url, reference), self.config.source_base)
            if local_path is not None:
                self.plan.add(local_path, dataset)

    async def discover(self) -> DiscoveryPlan:
        await self.discover_portal()
        await self.discover_laws()
        await self.discover_tsutatsu()
        await self.discover_beppyo()
        await self.discover_hanketsu()
        await self.discover_nta_qa()
        await self.discover_nta_guide()
        await self.discover_papers()
        await self.discover_treaties()
        validate_metrics(self.plan.metrics, self.config.minimum_counts)
        return self.plan

    async def discover_portal(self) -> None:
        await self.required_text("llms3.txt", "portal")
        await self.required_text("llms4.txt", "portal")

    async def discover_laws(self) -> None:
        root = "ai-law-db"
        entry_html = await self.required_text(f"{root}/index.html", "laws")
        self.add_html_references(f"{root}/index.html", entry_html, "laws")
        aliases = await self.required_json(f"{root}/data/law_aliases.json", "laws")
        await self.required_text(f"{root}/quickstart.txt", "laws")
        await self.optional_bytes(f"{root}/llms.txt", "laws")
        alias_map = aliases.get("aliases") if isinstance(aliases, dict) else None
        if not isinstance(alias_map, dict):
            raise DiscoveryError("ai-law-db law_aliases.json has no aliases mapping")
        codes = sorted({v for v in alias_map.values() if isinstance(v, str) and v})
        if not codes:
            raise DiscoveryError("ai-law-db did not expose any law codes")
        self.plan.metrics["law_codes"] = len(codes)
        article_total = 0
        for code in codes:
            index_path = f"{root}/enhanced/{code}/index.html"
            index_html = await self.required_text(index_path, "laws")
            self.add_html_references(index_path, index_html, "laws")
            article_ids = extract_article_ids(index_html)
            if not article_ids:
                raise DiscoveryError(f"law index has no article links: {index_path}")
            for article_id in article_ids:
                self.plan.add(f"{root}/enhanced/{code}/{article_id}.html", "laws")
                self.plan.add(f"{root}/text/{code}/{article_id}.txt", "laws")
            article_total += len(article_ids)
        self.plan.metrics["law_articles"] = article_total
        await self.discover_corporate_law_resolvers(root)

    async def discover_corporate_law_resolvers(self, root: str) -> None:
        index_path = f"{root}/data/resolve_meta_corp/index.json"
        index = await self.required_json(index_path, "laws")
        laws = index.get("laws") if isinstance(index, dict) else None
        if not isinstance(laws, dict):
            raise DiscoveryError("resolve_meta_corp index has no laws mapping")
        for law_code, metadata in laws.items():
            if not isinstance(metadata, dict):
                continue
            reference = metadata.get("bucket_index_url")
            if not isinstance(reference, str):
                continue
            law_index_path = resolve_dataset_reference(root, reference, self.config.source_base)
            if law_index_path is None:
                continue
            law_index = await self.required_json(law_index_path, "laws")
            buckets = law_index.get("buckets") if isinstance(law_index, dict) else None
            template = law_index.get("bucket_url_template") if isinstance(law_index, dict) else None
            if not isinstance(buckets, dict) or not isinstance(template, str):
                raise DiscoveryError(f"unsupported law resolver schema: {law_index_path}")
            for bucket in buckets:
                bucket_ref = template.replace("{law_code}", str(law_code)).replace(
                    "{bucket}", str(bucket)
                )
                bucket_path = resolve_dataset_reference(root, bucket_ref, self.config.source_base)
                if bucket_path is None:
                    raise DiscoveryError(f"law resolver escaped source: {bucket_ref}")
                await self.required_json(bucket_path, "laws")

    async def discover_tsutatsu(self) -> None:
        root = "ai-tsutatsu-db"
        entry_html = await self.required_text(f"{root}/index.html", "tsutatsu")
        self.add_html_references(f"{root}/index.html", entry_html, "tsutatsu")
        aliases = await self.required_json(f"{root}/data/doc_aliases.json", "tsutatsu")
        await self.required_text(f"{root}/llms.txt", "tsutatsu")
        await self.required_text(f"{root}/quickstart.txt", "tsutatsu")
        alias_map = aliases.get("aliases") if isinstance(aliases, dict) else None
        if not isinstance(alias_map, dict):
            raise DiscoveryError("ai-tsutatsu-db aliases are missing")
        codes = sorted({v for v in alias_map.values() if isinstance(v, str) and v})
        item_total = 0
        for code in codes:
            resolve_path = f"{root}/data/resolve_lite/{code}.json"
            resolve = await self.required_json(resolve_path, "tsutatsu")
            items = extract_item_ids(resolve.get("items") if isinstance(resolve, dict) else None)
            if not items:
                raise DiscoveryError(f"tsutatsu resolver has no items: {resolve_path}")
            index_url = resolve.get("index_url") if isinstance(resolve, dict) else None
            if isinstance(index_url, str):
                index_target = source_path_from_url(index_url, self.config.source_base)
                if index_target:
                    self.plan.add(index_target, "tsutatsu")
            for item_id in items:
                self.plan.add(f"{root}/text/{code}/{item_id}.txt", "tsutatsu")
                self.plan.add(f"{root}/enhanced/{code}/{item_id}.html", "tsutatsu")
            item_total += len(items)
        self.plan.metrics["tsutatsu_documents"] = len(codes)
        self.plan.metrics["tsutatsu_items"] = item_total

    async def discover_beppyo(self) -> None:
        root = "beppyo-db"
        llms = await self.required_text(f"{root}/llms.txt", "beppyo")
        for relative in (
            "quickstart.txt",
            "rel/rel-core.txt",
            "rel/rel-to-B04.txt",
            "rel/rel-from-B04.txt",
            "check/kenzan.txt",
            "flow/overview.txt",
            "lookup/article-map.txt",
        ):
            await self.optional_bytes(f"{root}/{relative}", "beppyo")
        ids = sorted(set(BEPPYO_ID_RE.findall(llms)))
        if not ids:
            raise DiscoveryError("beppyo-db llms.txt contains no form IDs")
        for item_id in ids:
            self.plan.add(f"{root}/beppyo/{item_id}.txt", "beppyo")
        self.plan.metrics["beppyo_items"] = len(ids)

    async def discover_hanketsu(self) -> None:
        base = "ai-hanketsu-db"
        shard_total = 0
        item_total = 0
        for sub_database in ("houjinzei", "saiketsu-houjinzei"):
            root = f"{base}/{sub_database}"
            await self.required_text(f"{root}/llms.txt", "hanketsu")
            await self.optional_bytes(f"{root}/quickstart.txt", "hanketsu")
            index_path = f"{root}/data/shards_index.json"
            index = await self.required_json(index_path, "hanketsu")
            shard_refs = shard_file_references(index)
            if not shard_refs:
                raise DiscoveryError(f"hanketsu shard index has no shards: {index_path}")
            database_items = 0
            for shard_ref in shard_refs:
                shard_path = resolve_dataset_reference(root, shard_ref, self.config.source_base)
                if shard_path is None:
                    raise DiscoveryError(f"invalid hanketsu shard path: {shard_ref}")
                shard_text = await self.required_text(shard_path, "hanketsu")
                rows = parse_tsv(shard_text, shard_path)
                for row in rows:
                    item_id = str(row.get("id") or row.get("case_id") or "").strip()
                    if not item_id:
                        raise DiscoveryError(f"hanketsu shard row has no id: {shard_path}")
                    self.plan.add(f"{root}/core/{item_id}.txt", "hanketsu")
                    database_items += 1
            declared = index.get("total") if isinstance(index, dict) else None
            if isinstance(declared, int) and declared != database_items:
                raise DiscoveryError(
                    f"hanketsu row count mismatch for {root}: declared={declared}, rows={database_items}"
                )
            shard_total += len(shard_refs)
            item_total += database_items
        self.plan.metrics["hanketsu_shards"] = shard_total
        self.plan.metrics["hanketsu_items"] = item_total

    async def discover_nta_qa(self) -> None:
        root = "ai-nta-qa-db"
        entry_html = await self.required_text(f"{root}/index.html", "nta_qa")
        self.add_html_references(f"{root}/index.html", entry_html, "nta_qa")
        index_path = f"{root}/data/resolve_lite/index.json"
        index = await self.required_json(index_path, "nta_qa")
        await self.required_text(f"{root}/quickstart.txt", "nta_qa")
        await self.optional_bytes(f"{root}/data/docs_index.tsv", "nta_qa")
        docs = index.get("docs") if isinstance(index, dict) else None
        if not isinstance(docs, dict):
            raise DiscoveryError("NTA QA resolver has no docs mapping")
        item_total = 0
        for doc_code, metadata in docs.items():
            if not isinstance(doc_code, str) or not isinstance(metadata, dict):
                raise DiscoveryError("invalid NTA QA document metadata")
            reference = metadata.get("resolve_lite_url") or metadata.get("url")
            if not isinstance(reference, str):
                raise DiscoveryError(f"NTA QA document has no resolver URL: {doc_code}")
            resolve_path = resolve_dataset_reference(root, reference, self.config.source_base)
            if resolve_path is None:
                raise DiscoveryError(f"NTA QA resolver escaped source: {reference}")
            resolve = await self.required_json(resolve_path, "nta_qa")
            items = extract_item_ids(resolve.get("items") if isinstance(resolve, dict) else None)
            if not items:
                raise DiscoveryError(f"NTA QA resolver has no items: {resolve_path}")
            index_url = resolve.get("index_url") if isinstance(resolve, dict) else None
            if isinstance(index_url, str):
                local_index = source_path_from_url(index_url, self.config.source_base)
                if local_index:
                    self.plan.add(local_index, "nta_qa")
            for item_id in items:
                self.plan.add(f"{root}/text/{doc_code}/{item_id}.txt", "nta_qa")
                self.plan.add(f"{root}/enhanced/{doc_code}/{item_id}.html", "nta_qa")
            item_total += len(items)
        shard_count, shard_rows, _ = await self.discover_sharded_assets(
            root,
            f"{root}/data/shards_index.json",
            "nta_qa",
            asset_columns=("url", "text_url"),
        )
        self.plan.metrics["nta_qa_documents"] = len(docs)
        self.plan.metrics["nta_qa_items"] = item_total
        self.plan.metrics["nta_qa_shards"] = shard_count
        self.plan.metrics["nta_qa_rows"] = shard_rows

    async def discover_nta_guide(self) -> None:
        root = "ai-nta-guide-db"
        entry_html = await self.required_text(f"{root}/index.html", "nta_guide")
        self.add_html_references(f"{root}/index.html", entry_html, "nta_guide")
        index_path = f"{root}/data/resolve_lite/index.json"
        index = await self.required_json(index_path, "nta_guide")
        await self.required_text(f"{root}/quickstart.txt", "nta_guide")
        docs = index.get("docs") if isinstance(index, dict) else None
        if not isinstance(docs, list) or not docs:
            raise DiscoveryError("NTA guide resolver has no docs list")
        total_items = 0
        total_parts = 0
        for document in docs:
            if not isinstance(document, dict):
                raise DiscoveryError("invalid NTA guide document metadata")
            doc_code = document.get("doc_code")
            reference = document.get("url") or document.get("file")
            if not isinstance(doc_code, str) or not isinstance(reference, str):
                raise DiscoveryError("NTA guide document lacks doc_code or resolver")
            resolve_path = resolve_dataset_reference(root, reference, self.config.source_base)
            if resolve_path is None:
                raise DiscoveryError(f"NTA guide resolver escaped source: {reference}")
            resolve = await self.required_json(resolve_path, "nta_guide")
            item_objects: list[Any] = []
            direct_items = resolve.get("items") if isinstance(resolve, dict) else None
            if isinstance(direct_items, list):
                item_objects.extend(direct_items)
            parts = resolve.get("parts") if isinstance(resolve, dict) else None
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict):
                        raise DiscoveryError(f"invalid NTA guide part in {resolve_path}")
                    part_ref = part.get("url") or part.get("file")
                    if not isinstance(part_ref, str):
                        raise DiscoveryError(f"NTA guide part has no path in {resolve_path}")
                    part_path = resolve_dataset_reference(root, part_ref, self.config.source_base)
                    if part_path is None:
                        raise DiscoveryError(f"NTA guide part escaped source: {part_ref}")
                    part_data = await self.required_json(part_path, "nta_guide")
                    part_items = part_data.get("items") if isinstance(part_data, dict) else None
                    if not isinstance(part_items, list):
                        raise DiscoveryError(f"NTA guide part has no items: {part_path}")
                    item_objects.extend(part_items)
                    total_parts += 1
            doc_item_ids: set[str] = set()
            for item in item_objects:
                if isinstance(item, dict):
                    item_id = item.get("item_id") or item.get("id")
                else:
                    item_id = item
                if not isinstance(item_id, (str, int)):
                    raise DiscoveryError(f"NTA guide item has no ID: {doc_code}")
                item_id = str(item_id).strip()
                if not item_id:
                    raise DiscoveryError(f"NTA guide item has an empty ID: {doc_code}")
                doc_item_ids.add(item_id)
                text_ref = item.get("text_url") if isinstance(item, dict) else None
                enhanced_ref = item.get("enhanced_url") if isinstance(item, dict) else None
                text_path = (
                    source_path_from_url(text_ref, self.config.source_base)
                    if isinstance(text_ref, str)
                    else None
                )
                enhanced_path = (
                    source_path_from_url(enhanced_ref, self.config.source_base)
                    if isinstance(enhanced_ref, str)
                    else None
                )
                self.plan.add(text_path or f"{root}/text/{doc_code}/{item_id}.txt", "nta_guide")
                self.plan.add(
                    enhanced_path or f"{root}/enhanced/{doc_code}/{item_id}.html",
                    "nta_guide",
                )
            declared = document.get("count")
            if isinstance(declared, int) and declared != len(doc_item_ids):
                raise DiscoveryError(
                    f"NTA guide item count mismatch for {doc_code}: "
                    f"declared={declared}, discovered={len(doc_item_ids)}"
                )
            index_target = f"{root}/enhanced/{doc_code}/index.html"
            await self.optional_bytes(index_target, "nta_guide")
            total_items += len(doc_item_ids)
        guide_shards, guide_rows, _ = await self.discover_sharded_assets(
            root,
            f"{root}/data/shards_index.json",
            "nta_guide",
            asset_columns=("text_url", "enhanced_url"),
        )
        self.plan.metrics["nta_guide_documents"] = len(docs)
        self.plan.metrics["nta_guide_parts"] = total_parts
        self.plan.metrics["nta_guide_items"] = total_items
        self.plan.metrics["nta_guide_shards"] = guide_shards
        self.plan.metrics["nta_guide_rows"] = guide_rows

    async def discover_papers(self) -> None:
        base = "ai-paper-db"
        await self.optional_bytes(f"{base}/llms.txt", "paper")
        total_shards = 0
        total_rows = 0
        pack_paths: set[str] = set()
        for sub_database in ("oecd-tpg-2022", "nta-tp-audit", "oecd-beps"):
            root = f"{base}/{sub_database}"
            await self.optional_bytes(f"{root}/llms.txt", "paper")
            await self.optional_bytes(f"{root}/quickstart.txt", "paper")
            shards, rows, assets = await self.discover_sharded_assets(
                root,
                f"{root}/data/shards_index.json",
                "paper",
                asset_columns=("core",),
            )
            total_shards += shards
            total_rows += rows
            pack_paths.update(path for path in assets if "/packs/" in f"/{path}")
            await self.discover_latin_terms(root, "paper")
        self.plan.metrics["paper_shards"] = total_shards
        self.plan.metrics["paper_rows"] = total_rows
        self.plan.metrics["paper_packs"] = len(pack_paths)

    async def discover_treaties(self) -> None:
        root = "ai-treaty-db/jp-tax-treaties"
        await self.required_text(f"{root}/data/docs_index.tsv", "treaty")
        await self.optional_bytes(f"{root}/quickstart.txt", "treaty")
        await self.optional_bytes(f"{root}/topics.txt", "treaty")
        shards, rows, assets = await self.discover_sharded_assets(
            root,
            f"{root}/data/shards_index.json",
            "treaty",
            asset_columns=("core",),
        )
        await self.discover_latin_terms(root, "treaty")
        self.plan.metrics["treaty_shards"] = shards
        self.plan.metrics["treaty_rows"] = rows
        self.plan.metrics["treaty_packs"] = len(
            {path for path in assets if "/packs/" in f"/{path}"}
        )

    async def discover_sharded_assets(
        self,
        root: str,
        index_path: str,
        dataset: str,
        *,
        asset_columns: tuple[str, ...],
    ) -> tuple[int, int, set[str]]:
        index = await self.required_json(index_path, dataset)
        shard_refs = shard_file_references(index)
        if not shard_refs:
            raise DiscoveryError(f"shard index has no shards: {index_path}")
        row_count = 0
        assets: set[str] = set()
        for reference in shard_refs:
            shard_path = resolve_dataset_reference(root, reference, self.config.source_base)
            if shard_path is None:
                raise DiscoveryError(f"invalid shard path in {index_path}: {reference}")
            shard_text = await self.required_text(shard_path, dataset)
            rows = parse_tsv(shard_text, shard_path)
            row_count += len(rows)
            for row in rows:
                for column in asset_columns:
                    value = row.get(column)
                    if not isinstance(value, str) or not value.strip():
                        continue
                    asset_path = local_asset_path(root, value.strip(), self.config.source_base)
                    if asset_path:
                        self.plan.add(asset_path, dataset)
                        assets.add(asset_path)
        return len(shard_refs), row_count, assets

    async def discover_latin_terms(self, root: str, dataset: str) -> None:
        index_path = f"{root}/data/latin_terms/index.json"
        index = await self.optional_json(index_path, dataset)
        if index is None:
            return
        for value in iter_strings(index):
            if not value.lower().endswith(".tsv"):
                continue
            path = resolve_dataset_reference(root, value, self.config.source_base)
            if path is None:
                raise DiscoveryError(f"invalid latin term path in {index_path}: {value}")
            self.plan.add(path, dataset)


def safe_relative_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("empty repository-relative path")
    decoded = unquote(path.strip()).replace("\\", "/")
    if decoded.startswith("/"):
        raise ValueError(f"absolute path is not allowed: {path}")
    parts = decoded.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe relative path: {path}")
    normalized = posixpath.normpath(decoded)
    if normalized == "." or normalized.startswith("../"):
        raise ValueError(f"unsafe relative path: {path}")
    return normalized


def source_url(source_base: str, path: str) -> str:
    clean = safe_relative_path(path)
    encoded = quote(clean, safe=URL_PATH_SAFE)
    return f"{source_base.rstrip('/')}/{encoded}"


def source_path_from_url(url: str, source_base: str) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    base = urlsplit(source_base.rstrip("/") + "/")
    resolved = urlsplit(urljoin(source_base.rstrip("/") + "/", url))
    if resolved.scheme != base.scheme or resolved.netloc != base.netloc:
        return None
    base_path = base.path.rstrip("/") + "/"
    if not resolved.path.startswith(base_path):
        return None
    relative = unquote(resolved.path[len(base_path) :])
    if not relative:
        return None
    return safe_relative_path(relative)


def resolve_dataset_reference(root: str, reference: str, source_base: str) -> str | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    reference = html.unescape(reference.strip())
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or reference.startswith("/"):
        return source_path_from_url(reference, source_base)
    joined = posixpath.join(safe_relative_path(root), reference)
    return safe_relative_path(joined)


def local_asset_path(root: str, value: str, source_base: str) -> str | None:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or value.startswith("/"):
        return source_path_from_url(value, source_base)
    clean = value.split("#", 1)[0].split("?", 1)[0].strip()
    if clean.startswith(LOCAL_ASSET_PREFIXES):
        return resolve_dataset_reference(root, clean, source_base)
    return None


def extract_article_ids(index_html: str) -> list[str]:
    return sorted(set(ARTICLE_LINK_RE.findall(index_html)), key=article_sort_key)


def article_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.split(r"[-:]", value))


def extract_item_ids(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value: Any = item
        if isinstance(item, dict):
            value = item.get("item_id") or item.get("id") or item.get("code")
        if isinstance(value, (str, int, float)):
            candidate = str(value).strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return result


def shard_file_references(index: Any) -> list[str]:
    if not isinstance(index, dict) or not isinstance(index.get("shards"), list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in index["shards"]:
        value: Any = item
        if isinstance(item, dict):
            value = item.get("file") or item.get("path") or item.get("url")
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def parse_tsv(text: str, path: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        if not reader.fieldnames:
            raise DiscoveryError(f"TSV has no header: {path}")
        return [dict(row) for row in reader]
    except csv.Error as exc:
        raise DiscoveryError(f"invalid TSV: {path}: {exc}") from exc


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def validate_metrics(metrics: Counter[str] | dict[str, int], minimums: dict[str, int]) -> None:
    failures = []
    for name, minimum in sorted(minimums.items()):
        actual = int(metrics.get(name, 0))
        if actual < minimum:
            failures.append(f"{name}: {actual} < required minimum {minimum}")
    if failures:
        raise DiscoveryError("source coverage check failed:\n- " + "\n- ".join(failures))


async def _create_discovery(config: Config, session: aiohttp.ClientSession) -> tuple[DiscoveryPlan, Fetcher]:
    fetcher = Fetcher(session, config)
    discovery = Discovery(config, fetcher)
    plan = await discovery.discover()
    return plan, fetcher


async def discover_mirror(config: Config) -> DiscoveryPlan:
    from .egov import inspect_egov_laws

    connector = aiohttp.TCPConnector(limit=max(config.concurrency * 2, 16))
    async with aiohttp.ClientSession(connector=connector) as session:
        plan, _ = await _create_discovery(config, session)
        plan.metrics.update(await inspect_egov_laws(config, session))
        return plan


async def download_targets(
    plan: DiscoveryPlan,
    fetcher: Fetcher,
    staging: Path,
) -> None:
    completed = 0
    completed_lock = asyncio.Lock()

    async def worker(target: Target) -> None:
        nonlocal completed
        body = fetcher.cache.get(target.path)
        if body is None:
            body = await fetcher.fetch_path(target.path, store_cache=False)
        if body is None:
            raise FetchError(f"required target unexpectedly missing: {target.path}")
        destination = staging / Path(target.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(destination.write_bytes, body)
        async with completed_lock:
            completed += 1
            if completed % 500 == 0 or completed == len(plan.targets):
                logging.info("Downloaded %s/%s files", completed, len(plan.targets))

    await asyncio.gather(*(worker(target) for target in plan.targets.values()))


def rewrite_source_urls(root: Path, config: Config) -> int:
    replacements = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise MirrorError(f"downloaded text file is not UTF-8: {path}") from exc
        count = text.count(config.source_base)
        if count:
            path.write_text(text.replace(config.source_base, config.mirror_base), encoding="utf-8")
            replacements += count
    return replacements


def generate_portal(root: Path, plan: DiscoveryPlan) -> None:
    links = [
        ("複雑な税務質問の作動テスト", "tax-question-tests/index.html"),
        ("AI向け総合案内", "llms3.txt"),
        ("AI向け詳細案内", "llms4.txt"),
        ("e-Gov公式法令（最新同期）", "egov-law-db/index.html"),
        ("e-Gov公式法令クイックスタート", "egov-law-db/quickstart.txt"),
        ("法令クイックスタート", "ai-law-db/quickstart.txt"),
        ("法令名一覧", "ai-law-db/data/law_aliases.json"),
        ("通達クイックスタート", "ai-tsutatsu-db/quickstart.txt"),
        ("別表一覧", "beppyo-db/llms.txt"),
        ("国税庁Q&Aクイックスタート", "ai-nta-qa-db/quickstart.txt"),
        ("国税庁手引きクイックスタート", "ai-nta-guide-db/quickstart.txt"),
        ("判決データ案内", "ai-hanketsu-db/houjinzei/llms.txt"),
        ("裁決データ案内", "ai-hanketsu-db/saiketsu-houjinzei/llms.txt"),
        ("租税条約クイックスタート", "ai-treaty-db/jp-tax-treaties/quickstart.txt"),
    ]
    available_links = [(label, href) for label, href in links if (root / href).is_file()]
    metric_rows = "\n".join(
        f"<tr><th>{html.escape(name)}</th><td>{value:,}</td></tr>"
        for name, value in sorted(plan.metrics.items())
    )
    link_rows = "\n".join(
        f'<li><a href="{html.escape(href, quote=True)}">{html.escape(label)}</a></li>'
        for label, href in available_links
    )
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jplawdb mirror</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.65}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.45rem;text-align:left}}th{{width:65%}}
code{{background:#f3f3f3;padding:.1rem .25rem}}
</style>
</head>
<body>
<h1>jplawdb mirror</h1>
<p>日本税法AIデータベースの検証済み静的ミラーです。公式な法令・税務判断の代替ではありません。</p>
<h2>入口</h2>
<ul>{link_rows}</ul>
<h2>今回の収録件数</h2>
<table><tbody>{metric_rows}</tbody></table>
<p>ファイル一覧とSHA-256は <a href="manifest.json">manifest.json</a> にあります。</p>
</body>
</html>
"""
    (root / "index.html").write_text(document, encoding="utf-8")
    (root / ".nojekyll").write_text("", encoding="utf-8")


def write_download_log(root: Path, plan: DiscoveryPlan) -> None:
    with (root / "download_log.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "dataset", "size", "sha256"])
        for target in sorted(plan.targets.values(), key=lambda item: item.path):
            path = root / target.path
            data = path.read_bytes()
            writer.writerow([target.path, target.dataset, len(data), hashlib.sha256(data).hexdigest()])


def generate_manifest(root: Path, config: Config, plan: DiscoveryPlan) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_base": config.source_base,
        "mirror_base": config.mirror_base,
        "metrics": dict(sorted(plan.metrics.items())),
        "files": entries,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _safe_sibling(path: Path, name: str) -> Path:
    parent = path.parent.resolve()
    candidate = (path.parent / name).resolve()
    if candidate.parent != parent:
        raise MirrorError(f"unsafe build path: {candidate}")
    return candidate


def _remove_tree(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise MirrorError(f"refusing to recursively remove non-directory: {path}")
        shutil.rmtree(path)


def atomic_publish(staging: Path, output: Path) -> None:
    output = output.resolve()
    backup = _safe_sibling(output, f".{output.name}.backup")
    _remove_tree(backup)
    moved_old = False
    try:
        if output.exists():
            if output.is_symlink() or not output.is_dir():
                raise MirrorError(f"output path is not a directory: {output}")
            output.rename(backup)
            moved_old = True
        staging.rename(output)
    except Exception:
        if moved_old and backup.exists() and not output.exists():
            backup.rename(output)
        raise
    _remove_tree(backup)


async def build_mirror(config: Config) -> BuildResult:
    from .egov import sync_egov_laws
    from .tax_questions import run_tax_question_tests
    from .verification import verify_output

    output = config.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _safe_sibling(output, f".{output.name}.staging")
    _remove_tree(staging)
    staging.mkdir(parents=True)
    try:
        connector = aiohttp.TCPConnector(limit=max(config.concurrency * 2, 16))
        async with aiohttp.ClientSession(connector=connector) as session:
            plan, fetcher = await _create_discovery(config, session)
            logging.info("Discovered %s targets", len(plan.targets))
            logging.info("Coverage metrics: %s", dict(sorted(plan.metrics.items())))
            egov_metrics = await sync_egov_laws(config, session, staging)
            plan.metrics.update(egov_metrics)
            logging.info("e-Gov metrics: %s", dict(sorted(egov_metrics.items())))
            await download_targets(plan, fetcher, staging)
        rewrites = rewrite_source_urls(staging, config)
        logging.info("Rewrote %s source URL occurrences", rewrites)
        question_metrics = await asyncio.to_thread(
            run_tax_question_tests,
            staging,
            config.mirror_base,
        )
        plan.metrics.update(question_metrics)
        logging.info("Tax-question metrics: %s", dict(sorted(question_metrics.items())))
        generate_portal(staging, plan)
        write_download_log(staging, plan)
        manifest = generate_manifest(staging, config, plan)
        report = verify_output(staging, config)
        atomic_publish(staging, output)
        return BuildResult(
            output_dir=output,
            file_count=report.file_count,
            total_bytes=report.total_bytes,
            metrics={str(k): int(v) for k, v in manifest["metrics"].items()},
        )
    except Exception:
        _remove_tree(staging)
        raise
