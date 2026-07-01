from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from jplawdb_mirror.core import (
    Config,
    DiscoveryPlan,
    NtaSourceSpec,
    generate_manifest,
)
from jplawdb_mirror.nta import (
    NtaError,
    _NtaResponse,
    _metrics,
    _write_output,
    decode_nta_html,
    parse_nta_document,
)
from jplawdb_mirror.verification import verify_output


HTML = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>No.100 テスト資料｜国税庁</title></head>
<body><div>OUTSIDE-NAVIGATION</div>
<div id="contents">
<h1>No.100 テスト資料</h1>
<p>[令和7年4月1日現在法令等]</p>
<p>REQUIRED-A <span>REQUIRED-B</span></p>
<script>SUPPRESSED-SCRIPT</script>
</div><footer>OUTSIDE-FOOTER</footer></body></html>
"""


def make_spec() -> NtaSourceSpec:
    return NtaSourceSpec(
        title="No.100 テスト資料",
        url="https://www.nta.go.jp/example.htm",
        required_terms=("REQUIRED-A", "REQUIRED-B"),
        require_legal_date=True,
        minimum_text_chars=30,
    )


def make_document():
    return parse_nta_document(
        _NtaResponse(
            body=HTML.encode("utf-8"),
            final_url="https://www.nta.go.jp/example.htm",
            content_type="text/html",
            last_modified="Mon, 25 May 2026 05:00:01 GMT",
            etag='"test"',
        ),
        code="test_source",
        spec=make_spec(),
        maximum_legal_age_days=550,
        today=date(2026, 7, 1),
    )


class NtaParserTest(unittest.TestCase):
    def test_extracts_only_contents_and_legal_date(self) -> None:
        document = make_document()
        self.assertEqual(document.legal_as_of, "2025-04-01")
        self.assertEqual(document.legal_age_days, 456)
        self.assertIn("REQUIRED-A REQUIRED-B", document.text)
        self.assertNotIn("OUTSIDE-NAVIGATION", document.text)
        self.assertNotIn("OUTSIDE-FOOTER", document.text)
        self.assertNotIn("SUPPRESSED-SCRIPT", document.text)

    def test_decodes_shift_jis_declared_in_meta(self) -> None:
        source = (
            '<html><head><meta charset="shift_jis"></head>'
            "<body><div id=\"contents\">法人税</div></body></html>"
        ).encode("cp932")
        decoded, declared, codec = decode_nta_html(source)
        self.assertIn("法人税", decoded)
        self.assertEqual(declared, "shift_jis")
        self.assertEqual(codec, "cp932")

    def test_rejects_stale_legal_date(self) -> None:
        old = HTML.replace("令和7年", "令和元年").encode("utf-8")
        with self.assertRaisesRegex(NtaError, "legal date is stale"):
            parse_nta_document(
                _NtaResponse(
                    body=old,
                    final_url="https://www.nta.go.jp/example.htm",
                    content_type="text/html",
                    last_modified=None,
                    etag=None,
                ),
                code="stale",
                spec=make_spec(),
                maximum_legal_age_days=550,
                today=date(2026, 7, 1),
            )


class NtaConfigTest(unittest.TestCase):
    def test_rejects_a_non_official_source_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                """
source_base: https://source.example
mirror_base: https://mirror.example
nta_official:
  sources:
    unsafe:
      title: Unsafe
      url: https://example.com/not-nta.htm
      required_terms: [term]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid NTA source specification"):
                Config.from_file(path)


class NtaOutputTest(unittest.TestCase):
    def test_generated_output_passes_full_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = make_document()
            config = Config(
                source_base="https://source.example",
                mirror_base="https://mirror.example",
                output_dir=root,
                nta_max_legal_age_days=5000,
                nta_sources={"test_source": make_spec()},
            )
            _write_output(
                root,
                [document],
                config.mirror_base,
                config.nta_max_legal_age_days,
            )
            (root / "index.html").write_text("<html></html>", encoding="utf-8")
            (root / ".nojekyll").write_text("", encoding="utf-8")
            (root / "download_log.tsv").write_text(
                "path\tdataset\tsize\tsha256\n",
                encoding="utf-8",
            )
            plan = DiscoveryPlan()
            plan.metrics.update(_metrics([document]))
            generate_manifest(root, config, plan)
            report = verify_output(root, config)
            self.assertGreater(report.file_count, 7)
            status = json.loads(
                (root / "nta-official-db" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
