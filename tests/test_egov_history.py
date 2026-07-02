from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from jplawdb_mirror.core import Config, DiscoveryPlan, generate_manifest
from jplawdb_mirror.egov import _metrics as current_metrics
from jplawdb_mirror.egov import _write_output as write_current_output
from jplawdb_mirror.egov_history import EgovHistory
from jplawdb_mirror.egov_history import _metrics as history_metrics
from jplawdb_mirror.egov_history import _write_output as write_history_output
from jplawdb_mirror.verification import verify_output
from test_egov import AS_OF, make_law


def revision_payload() -> dict[str, object]:
    return {
        "law_info": {
            "law_id": "340AC0000000034",
            "law_num": "昭和四十年法律第三十四号",
        },
        "revisions": [
            {
                "law_revision_id": "340AC0000000034_20240401_505AC0000000003",
                "amendment_enforcement_date": "2024-04-01",
                "current_revision_status": "Past",
                "amendment_law_num": "令和五年法律第三号",
                "amendment_law_title": "所得税法等の一部を改正する法律",
            },
            {
                "law_revision_id": "340AC0000000034_20250401_506AC0000000008",
                "amendment_enforcement_date": "2025-04-01",
                "current_revision_status": "Current",
                "amendment_law_num": "令和六年法律第八号",
                "amendment_law_title": "所得税法等の一部を改正する法律",
            },
        ],
    }


class EgovHistoryOutputTest(unittest.TestCase):
    def test_generated_history_passes_full_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = ("2025-04-01", "2024-04-01")
            current = make_law()
            snapshots = {
                as_of: (
                    replace(
                        current,
                        as_of=as_of,
                        law_revision_id=(
                            f"340AC0000000034_{as_of.replace('-', '')}"
                            "_TESTREVISION"
                        ),
                    ),
                )
                for as_of in dates
            }
            history = EgovHistory(
                snapshots=snapshots,
                revisions={"hojinzei": revision_payload()},
            )
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
                egov_history_dates=dates,
                egov_history_law_codes=("hojinzei",),
                egov_history_min_articles=6,
                egov_history_min_revisions=2,
            )
            write_current_output(root, AS_OF, [current], config.mirror_base)
            write_history_output(root, history, config.mirror_base)
            (root / "index.html").write_text("<html></html>", encoding="utf-8")
            (root / ".nojekyll").write_text("", encoding="utf-8")
            (root / "download_log.tsv").write_text(
                "path\tdataset\tsize\tsha256\n",
                encoding="utf-8",
            )
            plan = DiscoveryPlan()
            plan.metrics.update(current_metrics([current]))
            plan.metrics.update(history_metrics(history))
            generate_manifest(root, config, plan)

            report = verify_output(root, config)

            self.assertGreater(report.file_count, 25)
            status = json.loads(
                (root / "egov-law-db/history/status.json").read_text("utf-8")
            )
            self.assertEqual(status["article_count"], 6)
            self.assertEqual(status["revision_count"], 2)
            self.assertTrue(
                (
                    root
                    / "egov-law-db/history/2024-04-01/text/hojinzei/66-7.txt"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "egov-law-db/history/2025-04-01/"
                    "supplementary/hojinzei.txt"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
