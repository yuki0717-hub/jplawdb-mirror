from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from jplawdb_mirror.core import (
    Config,
    Discovery,
    DiscoveryPlan,
    extract_article_ids,
    extract_item_ids,
    generate_manifest,
    local_asset_path,
    resolve_dataset_reference,
    safe_relative_path,
    shard_file_references,
)
from jplawdb_mirror.verification import VerificationError, verify_output


SOURCE = "https://jplawdb.github.io/html-preview"


class DiscoveryHelpersTest(unittest.TestCase):
    def test_extracts_all_article_variants_from_index(self) -> None:
        source = '<a href="1.html">1</a><a href="66-7.html#p2">66-7</a><a href="index.html">top</a>'
        self.assertEqual(extract_article_ids(source), ["1", "66-7"])

    def test_extract_item_ids_supports_string_and_object_schemas(self) -> None:
        items = ["1-1", {"item_id": "2-1"}, {"id": 3}, "1-1"]
        self.assertEqual(extract_item_ids(items), ["1-1", "2-1", "3"])

    def test_shards_support_string_and_object_entries(self) -> None:
        index = {"shards": ["data/shards/a.txt", {"file": "data/shards/b.txt"}]}
        self.assertEqual(
            shard_file_references(index),
            ["data/shards/a.txt", "data/shards/b.txt"],
        )

    def test_dataset_relative_paths_are_not_duplicated(self) -> None:
        path = resolve_dataset_reference(
            "ai-paper-db/oecd-tpg-2022",
            "data/latin_terms/APA-00.tsv",
            SOURCE,
        )
        self.assertEqual(
            path,
            "ai-paper-db/oecd-tpg-2022/data/latin_terms/APA-00.tsv",
        )

    def test_absolute_source_url_is_mirrored_but_external_url_is_not(self) -> None:
        local = local_asset_path(
            "ai-nta-qa-db",
            f"{SOURCE}/ai-nta-qa-db/text/taxanswer_hojin/5100.txt",
            SOURCE,
        )
        external = local_asset_path(
            "ai-nta-qa-db",
            "https://www.nta.go.jp/example.pdf",
            SOURCE,
        )
        self.assertEqual(local, "ai-nta-qa-db/text/taxanswer_hojin/5100.txt")
        self.assertIsNone(external)

    def test_rejects_parent_traversal(self) -> None:
        with self.assertRaises(ValueError):
            safe_relative_path("data/../../secret.txt")


class SplitGuideDiscoveryTest(unittest.TestCase):
    def test_follows_parts_and_object_shards(self) -> None:
        base = SOURCE
        files = {
            "ai-nta-guide-db/index.html": '<a href="quickstart.txt">start</a>',
            "ai-nta-guide-db/quickstart.txt": "guide",
            "ai-nta-guide-db/data/resolve_lite/index.json": json.dumps(
                {
                    "docs": [
                        {
                            "doc_code": "guide_2026",
                            "count": 1,
                            "url": f"{base}/ai-nta-guide-db/data/resolve_lite/guide_2026.json",
                        }
                    ]
                }
            ),
            "ai-nta-guide-db/data/resolve_lite/guide_2026.json": json.dumps(
                {
                    "parts": [
                        {"file": "data/resolve_lite_parts/guide_2026/part-001.json"}
                    ]
                }
            ),
            "ai-nta-guide-db/data/resolve_lite_parts/guide_2026/part-001.json": json.dumps(
                {
                    "items": [
                        {
                            "item_id": "s001",
                            "text_url": f"{base}/ai-nta-guide-db/text/guide_2026/s001.txt",
                            "enhanced_url": f"{base}/ai-nta-guide-db/enhanced/guide_2026/s001.html",
                        }
                    ]
                }
            ),
            "ai-nta-guide-db/enhanced/guide_2026/index.html": '<a href="s001.html">item</a>',
            "ai-nta-guide-db/data/shards_index.json": json.dumps(
                {"shards": [{"file": "data/shards/shard-001.txt"}]}
            ),
            "ai-nta-guide-db/data/shards/shard-001.txt": (
                "doc_code\titem_id\ttext_url\tenhanced_url\n"
                f"guide_2026\ts001\t{base}/ai-nta-guide-db/text/guide_2026/s001.txt\t"
                f"{base}/ai-nta-guide-db/enhanced/guide_2026/s001.html\n"
            ),
        }

        class FakeFetcher:
            def __init__(self) -> None:
                self.cache: dict[str, bytes] = {}

            async def fetch_path(self, path: str, *, optional: bool = False, store_cache: bool = True):
                if path not in files:
                    if optional:
                        return None
                    raise AssertionError(f"unexpected required path: {path}")
                body = files[path].encode("utf-8")
                if store_cache:
                    self.cache[path] = body
                return body

        config = Config(source_base=SOURCE, mirror_base="https://example.test/mirror")
        discovery = Discovery(config, FakeFetcher())
        asyncio.run(discovery.discover_nta_guide())
        self.assertEqual(discovery.plan.metrics["nta_guide_documents"], 1)
        self.assertEqual(discovery.plan.metrics["nta_guide_parts"], 1)
        self.assertEqual(discovery.plan.metrics["nta_guide_items"], 1)
        self.assertEqual(discovery.plan.metrics["nta_guide_shards"], 1)
        self.assertIn("ai-nta-guide-db/text/guide_2026/s001.txt", discovery.plan.targets)
        self.assertIn("ai-nta-guide-db/enhanced/guide_2026/s001.html", discovery.plan.targets)


class VerificationTest(unittest.TestCase):
    def make_config(self, root: Path) -> Config:
        return Config(
            source_base=SOURCE,
            mirror_base="https://example.github.io/mirror",
            output_dir=root,
            minimum_counts={},
        )

    def prepare_generated_files(self, root: Path, index: str) -> Config:
        (root / "index.html").write_text(index, encoding="utf-8")
        (root / ".nojekyll").write_text("", encoding="utf-8")
        (root / "download_log.tsv").write_text("path\tdataset\tsize\tsha256\n", encoding="utf-8")
        config = self.make_config(root)
        generate_manifest(root, config, DiscoveryPlan())
        return config

    def test_zero_byte_files_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.prepare_generated_files(root, "<html></html>")
            report = verify_output(root, config)
            self.assertEqual(report.file_count, 3)

    def test_broken_relative_html_link_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.prepare_generated_files(root, '<a href="missing.txt">missing</a>')
            with self.assertRaisesRegex(VerificationError, "broken local link"):
                verify_output(root, config)

    def test_same_size_corruption_fails_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.prepare_generated_files(root, "<html>one</html>")
            (root / "index.html").write_text("<html>two</html>", encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "SHA-256 mismatch"):
                verify_output(root, config)

    def test_source_url_residue_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.prepare_generated_files(root, f"<p>{SOURCE}</p>")
            with self.assertRaisesRegex(VerificationError, "source URL was not rewritten"):
                verify_output(root, config)


if __name__ == "__main__":
    unittest.main()
