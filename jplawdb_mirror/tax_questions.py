from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .core import MirrorError


class TaxQuestionTestError(MirrorError):
    """A complex tax-question source bundle is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SourceCheck:
    path: str
    label: str
    required_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExternalSource:
    label: str
    url: str


@dataclass(frozen=True, slots=True)
class TaxQuestionScenario:
    id: str
    title: str
    question: str
    sources: tuple[SourceCheck, ...]
    external_sources: tuple[ExternalSource, ...] = ()


SCENARIOS = (
    TaxQuestionScenario(
        id="executive-compensation",
        title="役員給与の期中改定と事前確定届出給与",
        question=(
            "3月決算法人が業績悪化を理由に9月から代表取締役の月額報酬を減額し、"
            "12月に臨時賞与も支給する。各支給額を損金算入できる条件、届出期限、"
            "不相当に高額な部分の扱いを整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/hojinzei/34.txt",
                "法人税法34条",
                ("役員", "定期同額給与"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei_seirei/69.txt",
                "法人税法施行令69条",
                ("定期同額給与", "改定"),
            ),
            SourceCheck(
                "nta-official-db/text/executive_compensation_circular.txt",
                "国税庁公式・法人税基本通達9-2-12",
                ("定期同額給与", "非常勤役員"),
            ),
            SourceCheck(
                "nta-official-db/text/executive_compensation_taxanswer.txt",
                "国税庁公式・タックスアンサー5211",
                ("損金", "事前確定届出給与"),
            ),
            SourceCheck(
                "nta-official-db/text/executive_compensation_filing.txt",
                "国税庁公式・事前確定届出給与に関する届出",
                ("事前確定届出給与", "提出時期"),
            ),
        ),
        external_sources=(
            ExternalSource(
                "国税庁：事前確定届出給与に関する届出",
                "https://www.nta.go.jp/taxes/tetsuzuki/shinsei/annai/hojin/annai/5104.htm",
            ),
        ),
    ),
    TaxQuestionScenario(
        id="transfer-pricing",
        title="国外関連者への無形資産ライセンスと移転価格",
        question=(
            "日本法人が独自技術を海外子会社へライセンスするが、直接比較できる取引がない。"
            "最も適切な独立企業間価格の算定方法、比較可能性、無形資産への貢献、"
            "ローカルファイルで残すべき検討過程を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-4.txt",
                "租税特別措置法66条の4",
                ("国外関連者", "独立企業間価格"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu_seirei/39-12.txt",
                "租税特別措置法施行令39条の12",
                ("独立企業間価格", "比較対象取引"),
            ),
            SourceCheck(
                "nta-official-db/text/transfer_pricing_guidance.txt",
                "国税庁公式・移転価格事務運営要領",
                ("独立企業間価格", "ローカルファイル"),
            ),
            SourceCheck(
                "nta-official-db/text/transfer_pricing_methods.txt",
                "国税庁公式・独立企業間価格の算定上の留意点",
                ("最も適切な方法", "比較対象取引", "独立企業間価格"),
            ),
            SourceCheck(
                "ai-paper-db/nta-tp-audit/data/shards_index.json",
                "国税庁移転価格事務運営要領・参考事例集の索引",
            ),
            SourceCheck(
                "ai-paper-db/oecd-tpg-2022/data/shards_index.json",
                "OECD移転価格ガイドライン2022の索引",
            ),
        ),
        external_sources=(
            ExternalSource(
                "国税庁：移転価格事務運営要領",
                "https://www.nta.go.jp/law/jimu-unei/hojin/010601/00.htm",
            ),
        ),
    ),
    TaxQuestionScenario(
        id="invoice-input-tax-credit",
        title="インボイスがない取引の仕入税額控除",
        question=(
            "従業員の国内出張について、公共交通機関の領収書や適格請求書がない。"
            "仕入税額控除が認められる範囲、帳簿の追加記載事項、保存期間、"
            "経過措置の確認事項を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/shohizei/30.txt",
                "消費税法30条",
                ("課税仕入れ", "帳簿", "請求書"),
            ),
            SourceCheck(
                "egov-law-db/text/shohizei_seirei/49.txt",
                "消費税法施行令49条",
                ("帳簿", "請求書"),
            ),
            SourceCheck(
                "nta-official-db/text/invoice_taxanswer.txt",
                "国税庁公式・タックスアンサー6496",
                ("仕入税額控除", "適格請求書"),
            ),
            SourceCheck(
                "nta-official-db/text/consumption_tax_circular.txt",
                "国税庁公式・消費税基本通達11-6",
                ("出張旅費", "保存期間"),
            ),
        ),
        external_sources=(
            ExternalSource(
                "国税庁：インボイス制度Q&A",
                "https://www.nta.go.jp/taxes/shiraberu/zeimokubetsu/shohi/keigenzeiritsu/qa_01.htm",
            ),
        ),
    ),
    TaxQuestionScenario(
        id="revenue-recognition",
        title="複合契約の収益認識",
        question=(
            "SaaSの初期設定、導入支援、月額利用料を一つの契約で受け取る。"
            "法人税上の収益計上単位と時期を、履行義務、引渡し・役務完了、"
            "会計処理との相違を区別して整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/hojinzei/22-2.txt",
                "法人税法22条の2",
                ("収益", "引渡し"),
            ),
            SourceCheck(
                "nta-official-db/text/revenue_recognition_circular.txt",
                "国税庁公式・法人税基本通達2-1-1",
                ("個々の契約", "履行義務"),
            ),
        ),
        external_sources=(
            ExternalSource(
                "国税庁：収益認識に関する改正通達の趣旨説明",
                "https://www.nta.go.jp/law/joho-zeikaishaku/hojin/180530/index.htm",
            ),
        ),
    ),
    TaxQuestionScenario(
        id="unlisted-share-valuation",
        title="取引相場のない株式の相続税評価",
        question=(
            "非上場会社株式を相続した。会社規模、同族株主か少数株主か、"
            "類似業種比準・純資産価額・配当還元のどの方式を使うか、"
            "判定順序と追加で必要な事実を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/sozokuzei/22.txt",
                "相続税法22条",
                ("財産", "時価"),
            ),
            SourceCheck(
                "nta-official-db/text/unlisted_share_circular.txt",
                "国税庁公式・財産評価基本通達178",
                ("取引相場のない株式", "大会社"),
            ),
            SourceCheck(
                "nta-official-db/text/unlisted_share_taxanswer.txt",
                "国税庁公式・タックスアンサー4638",
                ("同族株主", "配当還元方式"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="additional-tax-penalties",
        title="修正申告・期限後申告と加算税",
        question=(
            "税務調査の連絡後に申告漏れへ気付き修正申告する。単純な計上漏れと"
            "証憑の改ざんが混在する場合、過少申告加算税、無申告加算税、"
            "重加算税の適用関係と『正当な理由』『隠蔽・仮装』を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/kokuzei_tsusoku/65.txt",
                "国税通則法65条",
                ("過少申告加算税", "正当な理由"),
            ),
            SourceCheck(
                "egov-law-db/text/kokuzei_tsusoku/66.txt",
                "国税通則法66条",
                ("無申告加算税", "正当な理由"),
            ),
            SourceCheck(
                "egov-law-db/text/kokuzei_tsusoku/68.txt",
                "国税通則法68条",
                ("重加算税", "隠蔽", "仮装"),
            ),
            SourceCheck(
                "nta-official-db/text/additional_tax_guidance.txt",
                "国税庁公式・加算税事務運営指針",
                ("過少申告加算税", "無申告加算税", "調査通知", "正当な理由"),
            ),
        ),
        external_sources=(
            ExternalSource(
                "国税庁：法人税の過少申告加算税及び無申告加算税の取扱い",
                "https://www.nta.go.jp/law/jimu-unei/hojin/100703_01/00.htm",
            ),
        ),
    ),
    TaxQuestionScenario(
        id="historical-invoice-start",
        title="インボイス制度開始日時点の仕入税額控除",
        question=(
            "2023年10月1日に行った国内課税仕入れについて、同日時点の消費税法を"
            "基準に、仕入税額控除に必要な帳簿・請求書等の要件を整理してください。"
            "現在法令で上書きせず、参照した過去基準日も回答に明記してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/history/2023-10-01/text/shohizei/30.txt",
                "2023年10月1日時点・消費税法30条",
                ("課税仕入れ", "帳簿", "請求書"),
            ),
            SourceCheck(
                "egov-law-db/history/2023-10-01/text/shohizei_seirei/49.txt",
                "2023年10月1日時点・消費税法施行令49条",
                ("帳簿", "請求書"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="historical-revenue-recognition",
        title="2024年度開始時点の収益認識",
        question=(
            "2024年4月1日に開始した事業年度のSaaS契約について、同日時点の"
            "法人税法22条・22条の2を基準に益金算入額と収益計上時期を整理して"
            "ください。現在法令との差異があれば区別してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/history/2024-04-01/text/hojinzei/22.txt",
                "2024年4月1日時点・法人税法22条",
                ("益金の額", "損金の額"),
            ),
            SourceCheck(
                "egov-law-db/history/2024-04-01/text/hojinzei/22-2.txt",
                "2024年4月1日時点・法人税法22条の2",
                ("収益", "引渡し"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="historical-executive-compensation",
        title="2025年度開始時点の役員給与",
        question=(
            "2025年4月1日に開始した事業年度中に役員報酬を改定する場合、同日時点の"
            "法人税法34条と施行令69条を基準に定期同額給与の要件を整理してください。"
            "取引日より後の改正を混在させないでください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/history/2025-04-01/text/hojinzei/34.txt",
                "2025年4月1日時点・法人税法34条",
                ("役員", "定期同額給与"),
            ),
            SourceCheck(
                "egov-law-db/history/2025-04-01/text/hojinzei_seirei/69.txt",
                "2025年4月1日時点・法人税法施行令69条",
                ("定期同額給与", "改定"),
            ),
        ),
    ),
)


BROKEN_TOPIC_GUIDES = (
    "ai-paper-db/nta-tp-audit/quickstart.txt",
    "ai-paper-db/oecd-tpg-2022/quickstart.txt",
)


def repair_broken_question_guides(root: Path) -> int:
    """Replace a known missing topic index with the available shard index."""

    repairs = 0
    for relative in BROKEN_TOPIC_GUIDES:
        path = root / relative
        missing_topic_index = path.parent / "data" / "topics.txt"
        available_shard_index = path.parent / "data" / "shards_index.json"
        if not path.is_file() or missing_topic_index.exists() or not available_shard_index.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        repaired, count = re.subn(
            r"(?m)^.*`data/topics\.txt`.*$",
            "- `data/shards_index.json`（章・節とshardの対応を確認する索引）",
            text,
        )
        if count:
            note = (
                "# mirror note: 元案内の未収録topic indexを、"
                "実在する shards_index.json に修正済み。\n"
            )
            path.write_text(note + repaired, encoding="utf-8")
            repairs += 1
    return repairs


def _date_marker(text: str) -> str | None:
    fetched_at: str | None = None
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith(("as_of:", "snapshot:", "[令和")):
            return stripped
        if stripped.startswith("legal_as_of:") and stripped.partition(":")[2].strip():
            return stripped
        if stripped.startswith("fetched_at:"):
            fetched_at = stripped
    return fetched_at


def _source_result(root: Path, source: SourceCheck, mirror_base: str) -> dict[str, Any]:
    path = root / source.path
    if not path.is_file():
        raise TaxQuestionTestError(f"tax-question source is missing: {source.path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TaxQuestionTestError(
            f"tax-question source is not readable UTF-8: {source.path}: {exc}"
        ) from exc
    if len(text.strip()) < 20:
        raise TaxQuestionTestError(f"tax-question source is unexpectedly empty: {source.path}")
    missing = [term for term in source.required_terms if term not in text]
    if missing:
        raise TaxQuestionTestError(
            f"tax-question source lost required terms: {source.path}: {missing}"
        )
    return {
        "label": source.label,
        "path": source.path,
        "url": f"{mirror_base.rstrip('/')}/{quote(source.path, safe='/')}",
        "required_terms": list(source.required_terms),
        "observed_date": _date_marker(text),
        "size": path.stat().st_size,
        "status": "passed",
    }


def _write_outputs(
    root: Path,
    mirror_base: str,
    scenario_results: list[dict[str, Any]],
    repairs: int,
) -> None:
    base = root / "tax-question-tests"
    base.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    source_checks = sum(len(item["sources"]) for item in scenario_results)
    report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "passed": True,
        "scenario_count": len(scenario_results),
        "source_check_count": source_checks,
        "navigation_repairs": repairs,
        "scenarios": scenario_results,
    }
    (base / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prompt_blocks = [
        "# 複雑な税務質問 作動テスト\n",
        "回答ルール:\n",
        "1. 質問文だけで結論を断定せず、追加で必要な事実を列挙する。\n",
        "2. 法律、施行令、通達、国税庁解説の順に根拠を区別する。\n",
        "3. 各ファイルの as_of / snapshot / 法令等現在日を回答に明記する。\n",
        "4. 結論、適用条件、反対の場合、実務対応、参照URLを分ける。\n",
        "5. 最終判断前に外部の公式資料も確認する。\n",
    ]
    for index, scenario in enumerate(scenario_results, start=1):
        prompt_blocks.extend(
            [
                f"\n## {index}. {scenario['title']}\n",
                f"{scenario['question']}\n",
                "参照必須:\n",
            ]
        )
        prompt_blocks.extend(
            f"- {source['label']}: {source['url']}\n" for source in scenario["sources"]
        )
        if scenario["external_sources"]:
            prompt_blocks.append("公式サイトで最終確認:\n")
            prompt_blocks.extend(
                f"- {source['label']}: {source['url']}\n"
                for source in scenario["external_sources"]
            )
    (base / "prompts.txt").write_text("".join(prompt_blocks), encoding="utf-8")

    sections: list[str] = []
    for index, scenario in enumerate(scenario_results, start=1):
        source_items = "".join(
            (
                f'<li><a href="../{html.escape(source["path"], quote=True)}">'
                f'{html.escape(source["label"])}</a>'
                f' — {html.escape(source["observed_date"] or "日付表示なし")}</li>'
            )
            for source in scenario["sources"]
        )
        external_items = "".join(
            (
                f'<li><a href="{html.escape(source["url"], quote=True)}">'
                f'{html.escape(source["label"])}</a></li>'
            )
            for source in scenario["external_sources"]
        )
        external_block = (
            f"<h3>公式サイトで最終確認</h3><ul>{external_items}</ul>"
            if external_items
            else ""
        )
        sections.append(
            f"<section><h2>{index}. {html.escape(scenario['title'])}</h2>"
            f"<p>{html.escape(scenario['question'])}</p>"
            f"<h3>ミラー内の確認済み資料</h3><ul>{source_items}</ul>"
            f"{external_block}</section>"
        )
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>複雑な税務質問 作動テスト</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.65}}
section{{border-top:1px solid #ccc;margin-top:2rem}}code{{background:#f3f3f3;padding:.1rem .25rem}}
</style></head><body>
<h1>複雑な税務質問 作動テスト</h1>
<p>全{len(scenario_results)}シナリオ、{source_checks}資料の存在と必須語をビルド時に検証済みです。</p>
<p><a href="prompts.txt">AI用テスト質問</a> /
<a href="report.json">機械可読テスト結果</a></p>
<p>このテストは回答の正しさ自体を保証しません。事実関係と基準日を確認し、最終判断は公式原文で行ってください。</p>
{''.join(sections)}
</body></html>
"""
    (base / "index.html").write_text(page, encoding="utf-8")


def run_tax_question_tests(
    root: Path,
    mirror_base: str,
    *,
    scenarios: tuple[TaxQuestionScenario, ...] = SCENARIOS,
) -> dict[str, int]:
    repairs = repair_broken_question_guides(root)
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        sources = [_source_result(root, source, mirror_base) for source in scenario.sources]
        results.append(
            {
                "id": scenario.id,
                "title": scenario.title,
                "question": scenario.question,
                "status": "passed",
                "sources": sources,
                "external_sources": [
                    {"label": source.label, "url": source.url}
                    for source in scenario.external_sources
                ],
            }
        )
    _write_outputs(root, mirror_base, results, repairs)
    return {
        "tax_question_scenarios": len(results),
        "tax_question_source_checks": sum(len(item["sources"]) for item in results),
        "tax_question_navigation_repairs": repairs,
    }
