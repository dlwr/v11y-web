# Claude Code ルール

## コミットルール

- 変更を行ったら必ずコミットすること
- コミットは小さく、意味のある単位で行う

## コード品質

- コードを書いたら必ず code-simplifier プラグインを使用してシンプル化すること
- 使い方: `/install code-simplifier` でインストール後、Task tool で `subagent_type: "code-simplifier"` を指定
- 注意: Codex（mcp__codex__codex）ではない！
