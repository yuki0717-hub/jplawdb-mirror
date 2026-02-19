#!/usr/bin/env python3
"""Mirror jplawdb html-preview content into a local output directory.

Phases:
A) Discover downloadable URLs from metadata endpoints.
B) Download all discovered URLs in parallel with retry/backoff.
C) Rewrite source-base URLs in downloaded text assets.
D) Generate index.html, .nojekyll, and manifest.json.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiohttp
import yaml

SOURCE_PREFIX = "https://jplawdb.github.io/html-preview"
TEXT_EXTENSIONS = {".txt", ".json", ".html", ".tsv"}


@dataclass(slots=True)
class Config:
    """Runtime configuration loaded from config.yaml."""

    source_base: str = SOURCE_PREFIX
    my_base: str = ""
    output_dir: str = "output"
    concurrency: int = 12
    delay_sec: float = 0.05
    max_retries: int = 4
    timeout_sec: int = 30


class RequestScheduler:
    """Ensure a minimum delay between request starts."""

    def __init__(self, delay_sec: float) -> None:
        self.delay_sec = max(0.0, delay_sec)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def wait_turn(self) -> None:
        """Sleep as needed so requests are spaced by configured delay."""
        if self.delay_sec <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait_for = self._next_allowed - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = time.monotonic()
            self._next_allowed = now + self.delay_sec


class MirrorContext:
    """Context/state used across all mirror phases."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.urls: set[str] = set()

    def normalize_source_base(self) -> str:
        """Return source_base without trailing slash."""
        return self.config.source_base.rstrip("/")

    def path_to_url(self, relative_path: str) -> str:
        """Convert a relative path under source_base to an absolute URL."""
        rel = relative_path.lstrip("/")
        return f"{self.normalize_source_base()}/{rel}"

    def url_to_relpath(self, url: str) -> str | None:
        """Convert URL to source-relative path if under source_base."""
        source = self.normalize_source_base()
        if not url.startswith(source):
            return None
        rel = url[len(source) :].lstrip("/")
        return rel or None

    def add_relative(self, relative_path: str) -> None:
        """Add a source-relative path to URL set."""
        self.urls.add(self.path_to_url(relative_path))

    def add_maybe_url(self, base_url: str, value: str) -> str:
        """Resolve an absolute/relative path-like value and store if in-scope."""
        resolved = urljoin(f"{base_url}", value)
        rel = self.url_to_relpath(resolved)
        if rel is None:
            logging.warning("Skip out-of-scope URL: %s (from %s)", value, base_url)
            return resolved
        self.add_relative(rel)
        return resolved


async def fetch_bytes(
    session: aiohttp.ClientSession,
    scheduler: RequestScheduler,
    url: str,
    config: Config,
    allow_404: bool = False,
) -> tuple[int, bytes | None]:
    """Fetch URL with retry/backoff. Return (status, body_or_none)."""
    for attempt in range(config.max_retries + 1):
        try:
            await scheduler.wait_turn()
            async with session.get(url, timeout=config.timeout_sec) as resp:
                status = resp.status
                if status == 404 and allow_404:
                    return status, None
                if 200 <= status < 300:
                    return status, await resp.read()
                if status == 404:
                    return status, None
                if status < 500 or attempt == config.max_retries:
                    logging.error("HTTP %s for %s", status, url)
                    return status, None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt == config.max_retries:
                logging.error("Request failed after retries: %s (%s)", url, exc)
                return -1, None
        await asyncio.sleep((2**attempt) * max(0.05, config.delay_sec))
    return -1, None


def parse_json_bytes(data: bytes | None, url: str) -> Any | None:
    """Parse JSON payload bytes safely."""
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logging.error("Invalid JSON at %s: %s", url, exc)
        return None


def parse_text_bytes(data: bytes | None, url: str) -> str | None:
    """Decode text payload bytes safely."""
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        logging.error("Invalid UTF-8 at %s: %s", url, exc)
        return None


def iter_strings(obj: Any) -> list[str]:
    """Extract all string leaves from nested JSON-like data."""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(iter_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(iter_strings(value))
    return out


def extract_item_ids(items: Any) -> list[str]:
    """Extract item IDs from resolve_lite items structures."""
    ids: list[str] = []
    if not isinstance(items, list):
        return ids
    for item in items:
        if isinstance(item, str):
            ids.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("item_id", "id", "itemCode", "code"):
            value = item.get(key)
            if isinstance(value, str) and value:
                ids.append(value)
                break
    return ids


async def phase_a_collect(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    """Phase A: gather all target URLs from metadata APIs/files."""
    await collect_a1_ai_law_db(ctx, session, scheduler)
    await collect_a2_ai_tsutatsu_db(ctx, session, scheduler)
    await collect_a3_beppyo_db(ctx, session, scheduler)
    await collect_a4_ai_hanketsu_db(ctx, session, scheduler)
    await collect_a5_ai_nta_qa_db(ctx, session, scheduler)
    await collect_a6_ai_nta_guide_db(ctx, session, scheduler)
    await collect_a7_ai_paper_db(ctx, session, scheduler)
    await collect_a8_ai_treaty_db(ctx, session, scheduler)
    ctx.add_relative("llms3.txt")


async def collect_a1_ai_law_db(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    base = "ai-law-db"
    index_rel = f"{base}/data/resolve_meta_corp/index.json"
    ctx.add_relative(f"{base}/data/law_aliases.json")
    ctx.add_relative(index_rel)
    ctx.add_relative(f"{base}/quickstart.txt")

    index_url = ctx.path_to_url(index_rel)
    _, data = await fetch_bytes(session, scheduler, index_url, ctx.config, allow_404=True)
    obj = parse_json_bytes(data, index_url)
    laws = obj.get("laws") if isinstance(obj, dict) else None
    if not isinstance(laws, dict):
        return

    for law_code, bucket_url_val in laws.items():
        if not isinstance(law_code, str) or not isinstance(bucket_url_val, str):
            continue
        bucket_url = ctx.add_maybe_url(index_url, bucket_url_val)
        _, bucket_data = await fetch_bytes(session, scheduler, bucket_url, ctx.config, allow_404=True)
        bucket_obj = parse_json_bytes(bucket_data, bucket_url)
        buckets = bucket_obj.get("buckets") if isinstance(bucket_obj, dict) else None
        article_ids: set[str] = set()
        if isinstance(buckets, dict):
            article_ids.update(str(k) for k in buckets.keys())
        elif isinstance(buckets, list):
            for item in buckets:
                if isinstance(item, str):
                    article_ids.add(item)
                elif isinstance(item, dict):
                    for key in ("article", "article_id", "id", "no"):
                        val = item.get(key)
                        if val is not None:
                            article_ids.add(str(val))
                            break
        for article in article_ids:
            ctx.add_relative(f"{base}/text/{law_code}/{article}.txt")
            ctx.add_relative(f"{base}/enhanced/{law_code}/{article}.html")


async def collect_a2_ai_tsutatsu_db(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    base = "ai-tsutatsu-db"
    alias_rel = f"{base}/data/doc_aliases.json"
    ctx.add_relative(alias_rel)
    ctx.add_relative(f"{base}/llms.txt")
    ctx.add_relative(f"{base}/quickstart.txt")

    alias_url = ctx.path_to_url(alias_rel)
    _, data = await fetch_bytes(session, scheduler, alias_url, ctx.config, allow_404=True)
    obj = parse_json_bytes(data, alias_url)
    if not isinstance(obj, dict):
        return

    doc_codes = sorted({str(v) for v in obj.values() if isinstance(v, str) and v})
    for doc_code in doc_codes:
        resolve_rel = f"{base}/data/resolve_lite/{doc_code}.json"
        ctx.add_relative(resolve_rel)
        resolve_url = ctx.path_to_url(resolve_rel)
        _, resolve_data = await fetch_bytes(session, scheduler, resolve_url, ctx.config, allow_404=True)
        resolve_obj = parse_json_bytes(resolve_data, resolve_url)
        items = resolve_obj.get("items") if isinstance(resolve_obj, dict) else None
        for item_id in extract_item_ids(items):
            ctx.add_relative(f"{base}/text/{doc_code}/{item_id}.txt")
            ctx.add_relative(f"{base}/enhanced/{doc_code}/{item_id}.html")


async def collect_a3_beppyo_db(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    base = "beppyo-db"
    extras = [
        f"{base}/llms.txt",
        f"{base}/rel/rel-core.txt",
        f"{base}/rel/rel-to-B04.txt",
        f"{base}/rel/rel-from-B04.txt",
        f"{base}/check/kenzan.txt",
        f"{base}/flow/overview.txt",
        f"{base}/lookup/article-map.txt",
    ]
    for rel in extras:
        ctx.add_relative(rel)

    llms_url = ctx.path_to_url(f"{base}/llms.txt")
    _, data = await fetch_bytes(session, scheduler, llms_url, ctx.config, allow_404=True)
    text = parse_text_bytes(data, llms_url) or ""
    ids = {m.group(1) for m in re.finditer(r"\|\s*(B\d+(?:-\d+)?)\s*\|", text)}
    for bid in ids:
        ctx.add_relative(f"{base}/beppyo/{bid}.txt")


async def collect_a4_ai_hanketsu_db(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    base = "ai-hanketsu-db"
    for sub_db in ["houjinzei", "saiketsu-houjinzei"]:
        prefix = f"{base}/{sub_db}"
        idx_rel = f"{prefix}/data/shards_index.json"
        ctx.add_relative(idx_rel)
        ctx.add_relative(f"{prefix}/llms.txt")
        ctx.add_relative(f"{prefix}/quickstart.txt")
        idx_url = ctx.path_to_url(idx_rel)
        _, idx_data = await fetch_bytes(session, scheduler, idx_url, ctx.config, allow_404=True)
        idx_obj = parse_json_bytes(idx_data, idx_url)
        shards = idx_obj.get("shards") if isinstance(idx_obj, dict) else None
        if not isinstance(shards, list):
            continue
        for shard in shards:
            if not isinstance(shard, str):
                continue
            shard_url = ctx.add_maybe_url(idx_url, shard)
            _, shard_data = await fetch_bytes(session, scheduler, shard_url, ctx.config, allow_404=True)
            shard_text = parse_text_bytes(shard_data, shard_url)
            if not shard_text:
                continue
            for line in shard_text.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                first = line.split("\t", 1)[0].strip()
                if not first or first.lower() in {"case_id", "id"}:
                    continue
                ctx.add_relative(f"{prefix}/core/{first}.txt")


async def collect_a5_ai_nta_qa_db(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    base = "ai-nta-qa-db"
    idx_rel = f"{base}/data/resolve_lite/index.json"
    ctx.add_relative(idx_rel)
    ctx.add_relative(f"{base}/data/shards_index.json")
    ctx.add_relative(f"{base}/quickstart.txt")
    ctx.add_relative(f"{base}/data/docs_index.tsv")

    idx_url = ctx.path_to_url(idx_rel)
    _, idx_data = await fetch_bytes(session, scheduler, idx_url, ctx.config, allow_404=True)
    idx_obj = parse_json_bytes(idx_data, idx_url)
    docs = idx_obj.get("docs") if isinstance(idx_obj, dict) else None
    if isinstance(docs, dict):
        for doc_code, resolve_url_val in docs.items():
            if not isinstance(doc_code, str) or not isinstance(resolve_url_val, str):
                continue
            resolve_url = ctx.add_maybe_url(idx_url, resolve_url_val)
            _, resolve_data = await fetch_bytes(session, scheduler, resolve_url, ctx.config, allow_404=True)
            resolve_obj = parse_json_bytes(resolve_data, resolve_url)
            items = resolve_obj.get("items") if isinstance(resolve_obj, dict) else None
            for item_id in extract_item_ids(items):
                ctx.add_relative(f"{base}/text/{doc_code}/{item_id}.txt")
                ctx.add_relative(f"{base}/enhanced/{doc_code}/{item_id}.html")

    await add_shards_and_optional_packs(ctx, session, scheduler, f"{base}/data/shards_index.json")


async def collect_a6_ai_nta_guide_db(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    base = "ai-nta-guide-db"
    idx_rel = f"{base}/data/resolve_lite/index.json"
    ctx.add_relative(idx_rel)
    ctx.add_relative(f"{base}/quickstart.txt")

    idx_url = ctx.path_to_url(idx_rel)
    _, idx_data = await fetch_bytes(session, scheduler, idx_url, ctx.config, allow_404=True)
    idx_obj = parse_json_bytes(idx_data, idx_url)
    docs = idx_obj.get("docs") if isinstance(idx_obj, dict) else None
    if not isinstance(docs, list):
        return
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_code = doc.get("doc_code")
        resolve_url_val = doc.get("url")
        if not isinstance(doc_code, str) or not isinstance(resolve_url_val, str):
            continue
        resolve_url = ctx.add_maybe_url(idx_url, resolve_url_val)
        _, resolve_data = await fetch_bytes(session, scheduler, resolve_url, ctx.config, allow_404=True)
        resolve_obj = parse_json_bytes(resolve_data, resolve_url)
        items = resolve_obj.get("items") if isinstance(resolve_obj, dict) else None
        for item_id in extract_item_ids(items):
            ctx.add_relative(f"{base}/text/{doc_code}/{item_id}.txt")
            ctx.add_relative(f"{base}/enhanced/{doc_code}/{item_id}.html")


async def collect_a7_ai_paper_db(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    base = "ai-paper-db"
    ctx.add_relative(f"{base}/llms.txt")
    for sub_db in ["oecd-tpg-2022", "nta-tp-audit", "oecd-beps"]:
        prefix = f"{base}/{sub_db}"
        ctx.add_relative(f"{prefix}/quickstart.txt")
        ctx.add_relative(f"{prefix}/llms.txt")
        idx_rel = f"{prefix}/data/shards_index.json"
        ctx.add_relative(idx_rel)
        idx_url = ctx.path_to_url(idx_rel)
        _, idx_data = await fetch_bytes(session, scheduler, idx_url, ctx.config, allow_404=True)
        idx_obj = parse_json_bytes(idx_data, idx_url)
        shards = idx_obj.get("shards") if isinstance(idx_obj, dict) else None
        if isinstance(shards, list):
            for shard in shards:
                if not isinstance(shard, str):
                    continue
                shard_url = ctx.add_maybe_url(idx_url, shard)
                _, shard_data = await fetch_bytes(session, scheduler, shard_url, ctx.config, allow_404=True)
                shard_text = parse_text_bytes(shard_data, shard_url)
                if not shard_text:
                    continue
                for line in shard_text.splitlines():
                    for field in line.split("\t"):
                        field = field.strip()
                        if "packs/" in field and not field.startswith("http"):
                            path = field[field.index("packs/") :]
                            ctx.add_relative(f"{prefix}/{path}")

        latin_idx_rel = f"{prefix}/data/latin_terms/index.json"
        latin_idx_url = ctx.path_to_url(latin_idx_rel)
        status, latin_idx_data = await fetch_bytes(session, scheduler, latin_idx_url, ctx.config, allow_404=True)
        if status != 404:
            ctx.add_relative(latin_idx_rel)
            latin_obj = parse_json_bytes(latin_idx_data, latin_idx_url)
            for s in iter_strings(latin_obj):
                if s.endswith(".tsv"):
                    if s.startswith("latin_terms/"):
                        ctx.add_relative(f"{prefix}/data/{s}")
                    else:
                        ctx.add_relative(f"{prefix}/data/latin_terms/{s}")


async def collect_a8_ai_treaty_db(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> None:
    prefix = "ai-treaty-db/jp-tax-treaties"
    docs_idx_rel = f"{prefix}/data/docs_index.tsv"
    shards_idx_rel = f"{prefix}/data/shards_index.json"
    for rel in [docs_idx_rel, shards_idx_rel, f"{prefix}/quickstart.txt", f"{prefix}/topics.txt"]:
        ctx.add_relative(rel)

    docs_idx_url = ctx.path_to_url(docs_idx_rel)
    _, docs_data = await fetch_bytes(session, scheduler, docs_idx_url, ctx.config, allow_404=True)
    docs_text = parse_text_bytes(docs_data, docs_idx_url)
    if docs_text:
        for line in docs_text.splitlines():
            if not line.strip() or line.lower().startswith("doc_id\t"):
                continue
            doc_id = line.split("\t", 1)[0].strip()
            if doc_id:
                ctx.add_relative(f"{prefix}/core/{doc_id}.txt")

    await add_shards_and_optional_packs(ctx, session, scheduler, shards_idx_rel)

    latin_idx_rel = f"{prefix}/data/latin_terms/index.json"
    latin_idx_url = ctx.path_to_url(latin_idx_rel)
    status, latin_idx_data = await fetch_bytes(session, scheduler, latin_idx_url, ctx.config, allow_404=True)
    if status != 404:
        ctx.add_relative(latin_idx_rel)
        latin_obj = parse_json_bytes(latin_idx_data, latin_idx_url)
        for s in iter_strings(latin_obj):
            if s.endswith(".tsv"):
                if s.startswith("latin_terms/"):
                    ctx.add_relative(f"{prefix}/data/{s}")
                else:
                    ctx.add_relative(f"{prefix}/data/latin_terms/{s}")


async def add_shards_and_optional_packs(
    ctx: MirrorContext,
    session: aiohttp.ClientSession,
    scheduler: RequestScheduler,
    shards_index_rel: str,
) -> None:
    """Add shard files and pack-like paths from shard TSVs."""
    idx_url = ctx.path_to_url(shards_index_rel)
    _, idx_data = await fetch_bytes(session, scheduler, idx_url, ctx.config, allow_404=True)
    idx_obj = parse_json_bytes(idx_data, idx_url)
    shards = idx_obj.get("shards") if isinstance(idx_obj, dict) else None
    if not isinstance(shards, list):
        return

    rel_root = shards_index_rel[: -len("data/shards_index.json")].rstrip("/")
    for shard in shards:
        if not isinstance(shard, str):
            continue
        shard_url = ctx.add_maybe_url(idx_url, shard)
        _, shard_data = await fetch_bytes(session, scheduler, shard_url, ctx.config, allow_404=True)
        shard_text = parse_text_bytes(shard_data, shard_url)
        if not shard_text:
            continue
        for line in shard_text.splitlines():
            for field in line.split("\t"):
                field = field.strip()
                if not field or field.startswith("http"):
                    continue
                if "packs/" in field:
                    path = field[field.index("packs/") :]
                    ctx.add_relative(f"{rel_root}/{path}")


async def phase_b_download_all(ctx: MirrorContext, session: aiohttp.ClientSession, scheduler: RequestScheduler) -> tuple[int, int, int]:
    """Phase B: download all discovered URLs."""
    output_dir = Path(ctx.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download_log.tsv"

    sem = asyncio.Semaphore(max(1, ctx.config.concurrency))
    rows: list[tuple[str, str, int, str]] = []
    success = missing = failed = 0

    async def worker(url: str) -> None:
        nonlocal success, missing, failed
        rel = ctx.url_to_relpath(url)
        if rel is None:
            return
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        async with sem:
            status, body = await fetch_bytes(session, scheduler, url, ctx.config, allow_404=True)

        if status == 404:
            missing += 1
            rows.append((url, "404", 0, str(out_path)))
            return
        if status < 0 or body is None:
            failed += 1
            rows.append((url, "ERROR", 0, str(out_path)))
            return

        out_path.write_bytes(body)
        size = len(body)
        success += 1
        rows.append((url, str(status), size, str(out_path)))

    await asyncio.gather(*(worker(url) for url in sorted(ctx.urls)))

    with log_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["url", "status", "size", "path"])
        writer.writerows(rows)

    logging.info("Download complete: ok=%s missing=%s failed=%s", success, missing, failed)
    return success, missing, failed


def phase_c_rewrite_urls(config: Config) -> int:
    """Phase C: replace source base URL with my_base in text-like files."""
    output_dir = Path(config.output_dir)
    if not config.my_base:
        logging.warning("config.my_base is empty; skipping rewrite")
        return 0

    old = SOURCE_PREFIX
    new = config.my_base.rstrip("/")
    rewrites = 0

    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        count = text.count(old)
        if count == 0:
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")
        rewrites += count

    logging.info("URL rewrites: %s", rewrites)
    return rewrites


def phase_d_generate(output_dir: Path) -> tuple[int, Path]:
    """Phase D: generate index.html, .nojekyll, and manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subdirs = sorted([p.name for p in output_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])

    index_html = output_dir / "index.html"
    links = "\n".join(f'<li><a href="{name}/">{name}</a></li>' for name in subdirs)
    index_html.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>jplawdb mirror</title></head><body>",
                "<h1>jplawdb mirror index</h1>",
                "<ul>",
                links,
                "</ul>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )

    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    manifest_entries: list[dict[str, Any]] = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "manifest.json"):
        rel = path.relative_to(output_dir).as_posix()
        data = path.read_bytes()
        manifest_entries.append(
            {
                "path": rel,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"generated_at": int(time.time()), "files": manifest_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Generated %s manifest entries", len(manifest_entries))
    return len(manifest_entries), manifest_path


def load_config(config_path: Path) -> Config:
    """Load config.yaml and apply defaults for missing keys."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a mapping")

    return Config(
        source_base=str(data.get("source_base", SOURCE_PREFIX)).rstrip("/"),
        my_base=str(data.get("my_base", "")).rstrip("/"),
        output_dir=str(data.get("output_dir", "output")),
        concurrency=int(data.get("concurrency", 12)),
        delay_sec=float(data.get("delay_sec", 0.05)),
        max_retries=int(data.get("max_retries", 4)),
        timeout_sec=int(data.get("timeout_sec", 30)),
    )


async def run(config: Config) -> int:
    """Execute all phases. Return process exit code."""
    logging.info("Starting mirror with config: %s", config)
    scheduler = RequestScheduler(config.delay_sec)
    connector = aiohttp.TCPConnector(limit=max(8, config.concurrency * 2))

    async with aiohttp.ClientSession(connector=connector) as session:
        ctx = MirrorContext(config)
        await phase_a_collect(ctx, session, scheduler)
        logging.info("Discovered %s URLs", len(ctx.urls))
        ok, missing, failed = await phase_b_download_all(ctx, session, scheduler)

    rewrites = phase_c_rewrite_urls(config)
    entries, manifest_path = phase_d_generate(Path(config.output_dir))

    logging.info(
        "Summary: discovered=%s downloaded=%s missing=%s failed=%s rewrites=%s manifest_entries=%s manifest=%s",
        len(ctx.urls),
        ok,
        missing,
        failed,
        rewrites,
        entries,
        manifest_path,
    )
    return 0


def parse_args() -> argparse.Namespace:
    """CLI options."""
    parser = argparse.ArgumentParser(description="Mirror jplawdb html-preview data")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")
    config = load_config(config_path)
    raise SystemExit(asyncio.run(run(config)))


if __name__ == "__main__":
    main()
