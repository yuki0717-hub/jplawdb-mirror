# jplawdb-mirror

## プロジェクト概要
日本税法AIデータベース (jplawdb.github.io/html-preview/) の完全独立ミラーを構築するツール。
元サイトが消滅しても単独動作する。

## 技術スタック
- Python 3.11+
- asyncio + aiohttp（非同期ダウンロード）
- GitHub Pages（ホスティング）

## テスト
- python verify.py で整合性チェック

## ファイル構成
- mirror.py: メインスクリプト（メタデータ収集→一括DL→URL書換→追加生成）
- verify.py: ダウンロード後の検証
- config.yaml: 設定
