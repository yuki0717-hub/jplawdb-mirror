# jplawdb-mirror

## 目的
日本税法AIデータベースの静的データを、完全性検証後にGitHub Pagesへ公開する。
「完全」は件数しきい値、必須URL、SHA-256、内部リンクで機械的に確認する。

## 構成
- jplawdb_mirror/core.py: 形式別の収集、クリーンビルド、原子的な入替え
- jplawdb_mirror/verification.py: manifest、ハッシュ、UTF-8、内部リンクの検証
- mirror.py: 収集CLI
- verify.py: 検証CLI
- config.yaml: URL、通信設定、最低件数
- tests/: ネットワーク不要の回帰テスト

## 必須確認
変更後は次を実行する。

    python -m unittest discover -s tests -v
    python mirror.py --plan

全量ビルドを実行できる場合は続けて次を実行する。

    python mirror.py
    python verify.py

## 禁止事項
- outputをGitへ追加しない
- 検証失敗を警告だけに変えない
- 404を成功扱いしない
- 既存ミラーを取得元にしない
- 元データ形式を確認せず最低件数を下げない
