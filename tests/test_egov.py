from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jplawdb_mirror.core import Config, DiscoveryPlan, generate_manifest
from jplawdb_mirror.egov import (
    EgovError,
    _write_output,
    parse_egov_xml,
    select_exact_law,
)
from jplawdb_mirror.verification import verify_output


AS_OF = "2026-06-30"
XML = """<?xml version="1.0" encoding="UTF-8"?>
<Law Lang="ja" LawType="Act">
  <LawNum>昭和四十年法律第三十四号</LawNum>
  <LawBody>
    <LawTitle>法人税法</LawTitle>
    <MainProvision>
      <Article Num="1">
        <ArticleCaption>（趣旨）</ArticleCaption>
        <ArticleTitle>第一条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphSentence><Sentence>この法律は法人税について定める。</Sentence></ParagraphSentence>
          <Item Num="1">
            <ItemTitle>一</ItemTitle>
            <ItemSentence><Sentence>ITEM-TEXT</Sentence></ItemSentence>
            <Subitem1 Num="1">
              <Subitem1Title>イ</Subitem1Title>
              <Subitem1Sentence><Sentence>SUBITEM-TEXT</Sentence></Subitem1Sentence>
            </Subitem1>
          </Item>
        </Paragraph>
      </Article>
      <Article Num="66_7">
        <ArticleTitle>第六十六条の七</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphSentence><Sentence>枝番号を含む条文。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
      <Article Num="70_2:70_4">
        <ArticleTitle>第七十条の二から第七十条の四まで</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphSentence><Sentence>範囲条文。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
    </MainProvision>
    <SupplProvision>
      <Article Num="1"><ArticleTitle>第一条</ArticleTitle></Article>
    </SupplProvision>
  </LawBody>
</Law>
""".encode()


def make_law():
    return parse_egov_xml(
        XML,
        code="hojinzei",
        expected_title="法人税法",
        law_type="Act",
        law_id="340AC0000000034",
        law_revision_id="340AC0000000034_20260401_507AC0000000013",
        law_num="昭和四十年法律第三十四号",
        updated="2026-04-01T10:41:41+09:00",
        amendment_enforcement_date="2026-04-01",
        as_of=AS_OF,
    )


class EgovParserTest(unittest.TestCase):
    def test_parses_main_articles_and_normalizes_branch_number(self) -> None:
        law = make_law()
        self.assertEqual(
            [article.key for article in law.articles],
            ["1", "66-7", "70-2-to-70-4"],
        )
        self.assertIn("枝番号を含む条文", law.articles[1].text)
        self.assertEqual(law.law_title, "法人税法")

    def test_preserves_article_paragraph_item_hierarchy(self) -> None:
        article = make_law().articles[0]
        self.assertIn("[a1:caption]", article.text)
        self.assertIn("[a1:title]", article.text)
        self.assertIn("[a1-p1:sentence]", article.text)
        self.assertIn("[a1-p1-i1:title] 一", article.text)
        self.assertIn("[a1-p1-i1:sentence] ITEM-TEXT", article.text)
        self.assertIn("[a1-p1-i1-s1-1:sentence] SUBITEM-TEXT", article.text)
        self.assertGreater(len(article.text.splitlines()), 5)

    def test_extracts_supplementary_provisions_separately(self) -> None:
        law = make_law()
        self.assertEqual(law.supplementary_count, 1)
        self.assertIn("[suppl001-a1:title] 第一条", law.supplementary_text)

    def test_rejects_title_mismatch(self) -> None:
        with self.assertRaisesRegex(EgovError, "title mismatch"):
            parse_egov_xml(
                XML,
                code="wrong",
                expected_title="所得税法",
                law_type="Act",
                law_id="id",
                law_revision_id="revision",
                law_num="昭和四十年法律第三十四号",
                updated="",
                amendment_enforcement_date=None,
                as_of=AS_OF,
            )

    def test_selects_only_exact_title_and_type(self) -> None:
        payload = {
            "laws": [
                {
                    "law_info": {"law_type": "Act", "law_id": "act"},
                    "revision_info": {"law_title": "法人税法"},
                },
                {
                    "law_info": {"law_type": "CabinetOrder", "law_id": "order"},
                    "revision_info": {"law_title": "法人税法施行令"},
                },
            ]
        }
        info, _ = select_exact_law(
            payload,
            code="hojinzei",
            title="法人税法",
            law_type="Act",
        )
        self.assertEqual(info["law_id"], "act")


class EgovOutputTest(unittest.TestCase):
    def test_generated_output_passes_full_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Config(
                source_base="https://source.example",
                mirror_base="https://mirror.example",
                output_dir=root,
                minimum_counts={},
                egov_api_base="https://laws.e-gov.go.jp/api/2",
                egov_as_of=AS_OF,
                egov_min_articles=3,
                egov_laws={
                    "hojinzei": {"title": "法人税法", "type": "Act"},
                },
            )
            _write_output(root, AS_OF, [make_law()], config.mirror_base)
            (root / "index.html").write_text("<html></html>", encoding="utf-8")
            (root / ".nojekyll").write_text("", encoding="utf-8")
            (root / "download_log.tsv").write_text(
                "path\tdataset\tsize\tsha256\n",
                encoding="utf-8",
            )
            plan = DiscoveryPlan()
            plan.metrics.update(
                {
                    "egov_law_codes": 1,
                    "egov_main_articles": 3,
                    "egov_supplementary_provisions": 1,
                    "egov_xml_bytes": len(XML),
                }
            )
            generate_manifest(root, config, plan)
            report = verify_output(root, config)
            self.assertGreater(report.file_count, 8)
            status = json.loads((root / "egov-law-db/status.json").read_text("utf-8"))
            self.assertEqual(status["article_count"], 3)
            self.assertEqual(status["supplementary_count"], 1)
            self.assertTrue(
                (root / "egov-law-db/supplementary/hojinzei.txt").is_file()
            )


if __name__ == "__main__":
    unittest.main()
