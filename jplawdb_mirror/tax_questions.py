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
        id="tp-intragroup-financing-guarantee",
        title="国外関連者への保証料・グループ内金融取引",
        question=(
            "日本親会社が海外子会社の銀行借入を保証し、保証料を受け取る。"
            "独立企業間価格として検討すべき保証料率、比較対象取引、"
            "信用補完の便益、移転価格文書化で残すべき分析を整理してください。"
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
                ("特殊の関係", "比較対象取引"),
            ),
            SourceCheck(
                "nta-official-db/text/transfer_pricing_guidance.txt",
                "国税庁公式・移転価格事務運営要領",
                ("独立企業間価格", "ローカルファイル"),
            ),
            SourceCheck(
                "nta-official-db/text/transfer_pricing_methods.txt",
                "国税庁公式・独立企業間価格の算定上の留意点",
                ("最も適切な方法", "比較対象取引"),
            ),
            SourceCheck(
                "ai-paper-db/oecd-tpg-2022/data/shards_index.json",
                "OECD移転価格ガイドライン2022の章・shard索引",
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
        id="tp-local-file-documentation",
        title="移転価格ローカルファイルと比較可能性分析",
        question=(
            "海外販売子会社との棚卸資産取引について、ローカルファイルを作成する。"
            "機能・リスク・資産分析、比較対象取引の選定、最も適切な方法、"
            "検証対象法人の選定と保存すべき資料を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-4.txt",
                "租税特別措置法66条の4",
                ("国外関連取引", "独立企業間価格"),
            ),
            SourceCheck(
                "nta-official-db/text/transfer_pricing_guidance.txt",
                "国税庁公式・移転価格事務運営要領",
                ("独立企業間価格", "ローカルファイル"),
            ),
            SourceCheck(
                "nta-official-db/text/transfer_pricing_methods.txt",
                "国税庁公式・独立企業間価格の算定上の留意点",
                ("最も適切な方法", "比較対象取引"),
            ),
            SourceCheck(
                "ai-paper-db/nta-tp-audit/data/shards_index.json",
                "国税庁移転価格事務運営要領・参考事例集の索引",
            ),
        ),
    ),
    TaxQuestionScenario(
        id="foreign-subsidiary-dividend-exemption",
        title="外国子会社配当の益金不算入と外国源泉税",
        question=(
            "日本法人が保有割合30%の外国子会社から配当を受ける。"
            "益金不算入の要件、保有期間、控除される費用相当額、"
            "損金算入対応配当や外国源泉税等の扱いを整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/hojinzei/23-2.txt",
                "法人税法23条の2",
                ("外国子会社", "益金の額に算入しない"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei_seirei/22-4.txt",
                "法人税法施行令22条の4",
                ("百分の二十五", "六月以上"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/39-2.txt",
                "法人税法39条の2",
                ("外国源泉税等", "損金の額に算入しない"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei_seirei/78-3.txt",
                "法人税法施行令78条の3",
                ("外国子会社", "外国法人税"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="foreign-tax-credit-limitation",
        title="外国税額控除の控除限度額と繰越",
        question=(
            "日本法人が海外支店所得に対して外国法人税を納付した。"
            "控除対象外国法人税、国外所得金額、控除限度額、"
            "控除限度超過額・控除余裕額の繰越関係を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/hojinzei/69.txt",
                "法人税法69条",
                ("外国税額の控除", "控除対象外国法人税"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei_seirei/141.txt",
                "法人税法施行令141条",
                ("外国法人税", "含まれない"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei_seirei/142.txt",
                "法人税法施行令142条",
                ("控除限度額", "調整国外所得金額"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei_seirei/145.txt",
                "法人税法施行令145条",
                ("繰越控除対象外国法人税額", "控除限度超過額"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="cfc-taxable-income",
        title="外国関係会社CFC税制の合算課税判定",
        question=(
            "日本法人が低税率国の持株会社を通じて海外事業を保有している。"
            "外国関係会社、特定外国関係会社、対象外国関係会社、"
            "課税対象金額、外国法人税のみなし控除、配当時の課税済金額を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-6.txt",
                "租税特別措置法66条の6",
                ("外国関係会社", "特定外国関係会社", "課税対象金額"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu_seirei/39-14.txt",
                "租税特別措置法施行令39条の14",
                ("課税対象金額", "請求権等勘案合算割合"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu_seirei/39-15.txt",
                "租税特別措置法施行令39条の15",
                ("適用対象金額", "基準により計算した金額"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-7.txt",
                "租税特別措置法66条の7",
                ("外国法人税", "控除対象外国法人税"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-8.txt",
                "租税特別措置法66条の8",
                ("特定課税対象金額", "課税済金額"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="cfc-passive-income",
        title="CFC税制の部分合算・受動的所得",
        question=(
            "海外子会社は実体のある事業会社だが、利子・配当・有価証券譲渡益も多い。"
            "部分対象外国関係会社、特定所得、部分課税対象金額、"
            "租税負担割合と実質支配関係の確認手順を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-6.txt",
                "租税特別措置法66条の6",
                ("部分対象外国関係会社", "特定所得の金額", "部分課税対象金額"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu_seirei/39-17-3.txt",
                "租税特別措置法施行令39条の17の3",
                ("部分適用対象金額", "剰余金の配当等"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu_seirei/39-17-2.txt",
                "租税特別措置法施行令39条の17の2",
                ("租税負担割合", "外国関係会社"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu_seirei/39-16.txt",
                "租税特別措置法施行令39条の16",
                ("実質支配関係", "特殊の関係"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="thin-capitalization",
        title="国外支配株主からの借入と過少資本税制",
        question=(
            "外国親会社から多額の借入をしている日本法人について、"
            "国外支配株主等、平均負債残高、資本持分、三倍基準、"
            "保証料や第三者借入を含めた損金不算入額の考え方を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-5.txt",
                "租税特別措置法66条の5",
                ("国外支配株主等", "平均負債残高", "損金の額に算入しない"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu_seirei/39-13.txt",
                "租税特別措置法施行令39条の13",
                ("国外支配株主等", "平均負債残高", "保証料"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-5-2.txt",
                "租税特別措置法66条の5の2",
                ("対象純支払利子等", "調整所得金額"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="earnings-stripping",
        title="過大支払利子税制と関連者支払利子",
        question=(
            "海外グループからの借入利子と第三者借入の保証料がある日本法人について、"
            "対象支払利子等、対象純支払利子等、調整所得金額の20%基準、"
            "関連者経由取引と過少資本税制との関係を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-5-2.txt",
                "租税特別措置法66条の5の2",
                ("対象純支払利子等", "調整所得金額", "損金の額に算入しない"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu_seirei/39-13-2.txt",
                "租税特別措置法施行令39条の13の2",
                ("対象純支払利子等", "関連者", "支払利子等"),
            ),
            SourceCheck(
                "egov-law-db/text/sozei_tokubetsu/66-5.txt",
                "租税特別措置法66条の5",
                ("国外支配株主等", "負債の利子等"),
            ),
        ),
    ),
    TaxQuestionScenario(
        id="royalty-withholding-foreign-corporation",
        title="外国法人へのロイヤルティ支払と源泉徴収",
        question=(
            "日本法人が外国法人へソフトウェア・特許技術の使用料を支払う。"
            "国内源泉所得該当性、外国法人への源泉徴収義務、"
            "租税条約で異なる定めがある場合の確認手順を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/shotokuzei/161.txt",
                "所得税法161条",
                ("使用料", "工業所有権", "著作権"),
            ),
            SourceCheck(
                "egov-law-db/text/shotokuzei/212.txt",
                "所得税法212条",
                ("外国法人", "源泉徴収", "翌月十日"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/138.txt",
                "法人税法138条",
                ("国内源泉所得", "外国法人", "恒久的施設"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/139.txt",
                "法人税法139条",
                ("租税条約", "国内源泉所得"),
            ),
            SourceCheck(
                "ai-treaty-db/jp-tax-treaties/quickstart.txt",
                "租税条約DBクイックスタート",
            ),
        ),
    ),
    TaxQuestionScenario(
        id="foreign-corporation-pe-filing",
        title="外国法人の恒久的施設と法人税申告",
        question=(
            "外国法人が日本に営業担当者と契約締結権限を持つ拠点を置く。"
            "恒久的施設の有無、国内源泉所得、外国法人の課税標準、"
            "確定申告・届出、租税条約で免税される場合の扱いを整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/hojinzei/138.txt",
                "法人税法138条",
                ("恒久的施設", "国内源泉所得"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/139.txt",
                "法人税法139条",
                ("租税条約", "内部取引"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/141.txt",
                "法人税法141条",
                ("恒久的施設を有する外国法人", "課税標準"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/144-6.txt",
                "法人税法144条の6",
                ("確定申告", "恒久的施設"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/149.txt",
                "法人税法149条",
                ("届出書", "恒久的施設"),
            ),
            SourceCheck(
                "ai-treaty-db/jp-tax-treaties/quickstart.txt",
                "租税条約DBクイックスタート",
            ),
        ),
    ),
    TaxQuestionScenario(
        id="pe-profit-attribution-and-foreign-tax-credit",
        title="PE帰属所得と外国法人の外国税額控除",
        question=(
            "外国法人の日本PEが国外でも関連する所得を得て外国法人税を負担した。"
            "PEに帰せられる国内源泉所得、租税条約上の内部取引、"
            "外国法人に係る外国税額控除と控除限度額を整理してください。"
        ),
        sources=(
            SourceCheck(
                "egov-law-db/text/hojinzei/138.txt",
                "法人税法138条",
                ("恒久的施設", "内部取引"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/139.txt",
                "法人税法139条",
                ("恒久的施設", "租税条約"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei/144-2.txt",
                "法人税法144条の2",
                ("外国法人税", "控除限度額"),
            ),
            SourceCheck(
                "egov-law-db/text/hojinzei_seirei/141.txt",
                "法人税法施行令141条",
                ("外国法人税", "含まれない"),
            ),
            SourceCheck(
                "ai-treaty-db/jp-tax-treaties/data/shards_index.json",
                "租税条約DB shard索引",
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


ANSWER_CHECKLISTS: dict[str, tuple[str, ...]] = {
    "transfer-pricing": (
        "国外関連者・国外関連取引・無形資産の範囲を最初に確定しているか。",
        "独立企業間価格の算定方法を、比較可能性・機能リスク資産分析・データ制約と結び付けて説明しているか。",
        "ローカルファイルに残すべき契約、取引実態、比較対象、レンジ、選定理由を列挙しているか。",
        "国内法・施行令・国税庁事務運営要領・OECD資料の位置付けを混同していないか。",
    ),
    "tp-intragroup-financing-guarantee": (
        "保証提供者・被保証者・貸手の関係と、保証料が国外関連取引に当たる前提を確認しているか。",
        "保証による信用補完効果、保証なし借入利率、保証あり借入利率、便益の有無を分けて検討しているか。",
        "直接比較可能な保証取引がない場合の方法選択と比較可能性調整を説明しているか。",
        "契約書、信用格付け、借入条件、第三者保証料、グループ内金融方針をローカルファイルの証拠として示しているか。",
    ),
    "tp-local-file-documentation": (
        "対象取引、国外関連者、事業年度、取引金額、検証対象法人を特定しているか。",
        "機能・リスク・資産分析と比較対象取引又は比較対象法人の選定理由を明示しているか。",
        "最も適切な方法、利益水準指標、比較対象レンジ、差異調整、除外基準を説明しているか。",
        "保存・提出対応、推定課税や同業者調査リスク、後日の資料更新方針に触れているか。",
    ),
    "foreign-subsidiary-dividend-exemption": (
        "外国子会社該当性、保有割合、保有期間、配当の性質を確認しているか。",
        "益金不算入割合と、費用相当額・外国源泉税の損金算入又は税額控除の可否を分けて説明しているか。",
        "外国税額控除との重複排除、会計上の受取配当・源泉税処理との税務調整を区別しているか。",
        "租税条約で配当源泉税率が変わる場合は、条約確認を別途必要事項として残しているか。",
    ),
    "foreign-tax-credit-limitation": (
        "控除対象外国法人税か、損金算入対象か、租税条約上の軽減・免除後の税額かを区別しているか。",
        "国外所得金額、全世界所得、法人税額、控除限度額の関係を計算順序として説明しているか。",
        "控除限度超過額・控除余裕額の繰越、地方税側の扱い、資料保存を確認しているか。",
        "外国税額控除だけで結論を出さず、外国子会社配当益金不算入やPE帰属所得との関係を確認しているか。",
    ),
    "cfc-taxable-income": (
        "外国関係会社、特定外国関係会社、対象外国関係会社等の判定順序を示しているか。",
        "持株・支配関係、租税負担割合、経済活動基準、事業実体を事実確認項目として列挙しているか。",
        "会社単位の合算課税、部分合算、適用除外を混同せず、どの段階の判定かを明示しているか。",
        "課税対象金額、外国法人税のみなし控除、二重課税調整、別表・資料保存に触れているか。",
    ),
    "cfc-passive-income": (
        "部分対象外国関係会社か、特定外国関係会社等として会社単位合算に進むのかを先に確認しているか。",
        "利子、配当、有価証券譲渡益、デリバティブ等の受動的所得を種類別に整理しているか。",
        "特定所得から除外される事業関連所得や実体ある所得の可能性を確認しているか。",
        "実質支配、関連者取引、租税負担割合、資料保存を結論の前提として明記しているか。",
    ),
    "thin-capitalization": (
        "国外支配株主等、資金供与者等、第三者を介した借入又は保証の有無を確認しているか。",
        "平均負債残高、自己資本持分、利子等の範囲を計算要素として分けているか。",
        "過少資本税制による損金不算入額と、過大支払利子税制との適用関係を整理しているか。",
        "独立企業間価格としての利率・保証料の検討と、資本構成規制の検討を混同していないか。",
    ),
    "earnings-stripping": (
        "対象純支払利子等、対象支払利子等、控除対象受取利子等、調整所得金額を分けているか。",
        "20%基準、適用除外、関連者支払利子の範囲、第三者借入の扱いを確認しているか。",
        "損金不算入額の繰越、過少資本税制との関係、移転価格税制による利率検証を併記しているか。",
        "連結・グループ内金融・保証料がある場合の追加資料を確認事項として残しているか。",
    ),
    "royalty-withholding-foreign-corporation": (
        "ロイヤルティが国内源泉所得に当たる根拠と、使用地・権利内容・支払者を確認しているか。",
        "外国法人への支払時の源泉徴収義務、税率、納付時期、グロスアップ条項を確認しているか。",
        "租税条約による軽減・免除、特典制限、受益者、居住者証明書等の手続を別項目にしているか。",
        "PEがある場合の法人税課税・源泉徴収との関係を、PEなしの場合と分けて説明しているか。",
    ),
    "foreign-corporation-pe-filing": (
        "支店、建設PE、代理人PE、準備的補助的活動のいずれの論点かを特定しているか。",
        "国内源泉所得とPE帰属所得を区別し、外国法人の課税標準と申告義務に結び付けているか。",
        "法人税申告、帳簿保存、納税管理人、源泉徴収との関係を実務対応として示しているか。",
        "国内法のPE判定と租税条約上のPE判定が異なる可能性を明示しているか。",
    ),
    "pe-profit-attribution-and-foreign-tax-credit": (
        "PE帰属所得、内部取引、機能・リスク・資産の帰属を分けて説明しているか。",
        "外国法人側の外国税額控除で対象になる税額と控除限度額の前提を確認しているか。",
        "国内源泉所得、国外源泉所得、PEに帰せられる所得の区分を混同していないか。",
        "租税条約の事業利得条項、二重課税排除条項、国内法の申告計算の順に確認しているか。",
    ),
}


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
    answer_checklist_count = sum(
        len(item.get("answer_checklist", ())) for item in scenario_results
    )
    report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "passed": True,
        "scenario_count": len(scenario_results),
        "source_check_count": source_checks,
        "answer_checklist_count": answer_checklist_count,
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
            ]
        )
        if scenario["answer_checklist"]:
            prompt_blocks.append("回答チェック観点:\n")
            prompt_blocks.extend(
                f"- {item}\n" for item in scenario["answer_checklist"]
            )
        prompt_blocks.append("参照必須:\n")
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
        checklist_items = "".join(
            f"<li>{html.escape(item)}</li>"
            for item in scenario.get("answer_checklist", ())
        )
        checklist_block = (
            f"<h3>AI回答チェック観点</h3><ul>{checklist_items}</ul>"
            if checklist_items
            else ""
        )
        sections.append(
            f"<section><h2>{index}. {html.escape(scenario['title'])}</h2>"
            f"<p>{html.escape(scenario['question'])}</p>"
            f"{checklist_block}"
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
                "answer_checklist": list(ANSWER_CHECKLISTS.get(scenario.id, ())),
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
        "tax_question_answer_check_items": sum(
            len(item["answer_checklist"]) for item in results
        ),
        "tax_question_navigation_repairs": repairs,
    }
