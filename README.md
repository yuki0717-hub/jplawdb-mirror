# jplawdb-mirror

日本税法AIデータベースの静的データを、検証してからGitHub Pagesへ公開するミラービルダーです。
一般向けの検索サイトやAIチャットではなく、AIエージェントやプログラムが条文・通達・Q&A等を安定したURLで取得するためのデータ配信サイトです。

## 収録対象

- 法令24種（法・施行令・施行規則等）
- 税務通達
- 法人税別表
- 国税庁Q&A・手引き
- 税務判決・裁決
- OECD等の税務文献
- 日本の租税条約

各データセット固有の索引形式を専用アダプターで解釈します。形式が変わった場合、推測で公開を続けずビルドを失敗させます。

## e-Gov公式法令同期

法令24種は、既存のAI向け法令データに加えて、e-Gov法令API Version 2から毎回直接取得します。
公式XMLと本則の条文別テキストを `egov-law-db` に生成します。

- `egov-law-db/index.html`: 人向け法令一覧
- `egov-law-db/index.json`: プログラム向け索引
- `egov-law-db/status.json`: 法令ID、法令履歴ID、基準日、XMLハッシュ
- `egov-law-db/xml/{law_code}.xml`: e-Gov公式XML
- `egov-law-db/text/{law_code}/{article}.txt`: AI向け条文テキスト

税務質問では、更新日が固定された旧 `ai-law-db` より `egov-law-db` を優先してください。
e-Gov APIで法令名が一意に解決できない、24法令のいずれかを取得できない、XMLを解析できない、最低条文数を下回る場合は公開しません。

## 複雑な税務質問の作動テスト

毎回のビルドで、役員給与、移転価格、仕入税額控除、収益認識、非上場株式評価、加算税の6シナリオを検査します。
法律・施行令・通達・国税庁解説など19資料について、ファイルの存在、UTF-8、必須語、参照リンクを確認します。

- `tax-question-tests/index.html`: 人向けのテスト結果と参照資料
- `tax-question-tests/prompts.txt`: AIへそのまま渡せる質問と回答ルール
- `tax-question-tests/report.json`: ビルド時の機械可読テスト結果

e-Gov条文テキストは条・項・号の位置を保持するマーカー付きで生成します。
テストは回答の正しさそのものを保証しないため、最終判断では各シナリオに掲載した公式原文も確認してください。

## 安全設計

- 毎回空のステージングディレクトリから構築
- 必須ファイルの404・不明なメタデータ形式・最低件数割れで停止
- e-Gov法令24種の法令ID・履歴ID・基準日・XMLを検証
- ファイルサイズとSHA-256をmanifest.jsonで検証
- HTMLの相対リンク切れを検証
- 元サイトURLの書換え漏れを検証
- 全検証成功後だけoutputを入れ替え
- outputはGit管理しないため、古い生成物が混ざらない

## 実行

Python 3.11以上を使用します。

    python -m venv .venv
    .venv/Scripts/pip install -r requirements.txt
    .venv/Scripts/python mirror.py --plan
    .venv/Scripts/python mirror.py
    .venv/Scripts/python verify.py

Linux/macOSでは .venv/bin/python と .venv/bin/pip を使用してください。

mirror.py --plan は全量を保存せず、元データの索引形式と収録予定件数を検査します。

## テスト

    python -m unittest discover -s tests -v

テストは、過去に問題となった次の形式を固定して確認します。

- 文字列形式とオブジェクト形式のshard
- 分割された国税庁手引き索引
- 法令の枝番条文
- e-Gov XMLの法令名一致、本則抽出、枝番号変換
- e-Gov同期状態と生成ファイル数の整合
- latin_termsの相対パス
- ゼロバイトファイル
- 同一サイズの改ざん
- HTMLリンク切れ

## 更新と公開

GitHub Actionsは毎週月曜日と手動実行に対応します。ビルドまたは検証が失敗した場合、既存のgh-pagesは更新されません。
最低件数はconfig.yamlにあります。値を下げる場合は、元サイトで本当に資料が削除されたことを確認してください。

## 注意

このリポジトリと公開サイトは政府の公式サービスではありません。税務判断にはe-Gov、国税庁、財務省等の最新原文を確認してください。
ミラー対象資料の著作権・利用条件は各原資料に従います。特にOECD資料等を再配布する場合は、権利と出典表示を別途確認してください。
