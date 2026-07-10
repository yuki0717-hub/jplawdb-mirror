from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jplawdb_mirror.core import Config, DiscoveryPlan, generate_manifest
from jplawdb_mirror.tax_questions import (
    SCENARIOS,
    SourceCheck,
    TaxQuestionScenario,
    TaxQuestionTestError,
    repair_broken_question_guides,
    run_tax_question_tests,
)
from jplawdb_mirror.verification import verify_output


class TaxQuestionTestRunnerTest(unittest.TestCase):
    def test_production_scenarios_prioritize_direct_nta_sources(self) -> None:
        sources = [source for scenario in SCENARIOS for source in scenario.sources]
        official = [
            source for source in sources if source.path.startswith("nta-official-db/")
        ]
        self.assertEqual(len(SCENARIOS), 20)
        self.assertEqual(len(sources), 78)
        self.assertEqual(len(official), 15)
        self.assertFalse(
            any(
                source.path.startswith(("ai-nta-qa-db/", "ai-tsutatsu-db/"))
                for source in sources
            )
        )

    def scenario(self) -> tuple[TaxQuestionScenario, ...]:
        return (
            TaxQuestionScenario(
                id="sample",
                title="Sample",
                question="Check the sources.",
                sources=(
                    SourceCheck(
                        "sources/article.txt",
                        "Article",
                        ("REQUIRED-A", "REQUIRED-B"),
                    ),
                ),
            ),
        )

    def test_generates_a_machine_readable_pass_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources" / "article.txt"
            source.parent.mkdir(parents=True)
            source.write_text(
                "as_of: 2026-06-30\nREQUIRED-A and REQUIRED-B are present.\n",
                encoding="utf-8",
            )
            metrics = run_tax_question_tests(
                root,
                "https://example.test/mirror",
                scenarios=self.scenario(),
            )
            report = json.loads(
                (root / "tax-question-tests" / "report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["scenario_count"], 1)
            self.assertEqual(report["source_check_count"], 1)
            self.assertEqual(metrics["tax_question_scenarios"], 1)
            self.assertEqual(
                report["scenarios"][0]["sources"][0]["observed_date"],
                "as_of: 2026-06-30",
            )
            (root / "index.html").write_text(
                '<a href="tax-question-tests/index.html">tests</a>',
                encoding="utf-8",
            )
            (root / ".nojekyll").write_text("", encoding="utf-8")
            (root / "download_log.tsv").write_text(
                "path\tdataset\tsize\tsha256\n",
                encoding="utf-8",
            )
            config = Config(
                source_base="https://source.example",
                mirror_base="https://example.test/mirror",
                output_dir=root,
            )
            plan = DiscoveryPlan()
            plan.metrics.update(metrics)
            generate_manifest(root, config, plan)
            self.assertGreater(verify_output(root, config).html_links_checked, 1)

    def test_missing_required_term_stops_the_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources" / "article.txt"
            source.parent.mkdir(parents=True)
            source.write_text("This has REQUIRED-A only.", encoding="utf-8")
            with self.assertRaisesRegex(TaxQuestionTestError, "REQUIRED-B"):
                run_tax_question_tests(
                    root,
                    "https://example.test/mirror",
                    scenarios=self.scenario(),
                )

    def test_repairs_a_missing_topic_index_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "ai-paper-db" / "nta-tp-audit"
            (dataset / "data").mkdir(parents=True)
            (dataset / "data" / "shards_index.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            quickstart = dataset / "quickstart.txt"
            quickstart.write_text(
                "入口\n- `data/topics.txt`（テーマ別索引）\n",
                encoding="utf-8",
            )
            self.assertEqual(repair_broken_question_guides(root), 1)
            repaired = quickstart.read_text(encoding="utf-8")
            self.assertNotIn("data/topics.txt", repaired)
            self.assertIn("data/shards_index.json", repaired)


if __name__ == "__main__":
    unittest.main()
