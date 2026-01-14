# v11n-web 開発履歴

## 2025-01-14: 初期実装

### 完了タスク

- プロジェクト作成 (`ghq create v11n-web`)
- Vite + React + TypeScript セットアップ
- Tailwind CSS 導入 (@tailwindcss/vite)
- ONNX Runtime Web 導入
- NSNet2モデル (nsnet2-48k.onnx) をv11yからコピー
- 録音フック (useRecorder.ts) 実装
  - WebAudio API による48kHz録音
  - リアルタイム振幅取得
  - 一時停止/再開対応
- ノイズ除去 (noiseReducer.ts) 実装
  - Cooley-Tukey FFT/IFFT
  - STFT/ISTFT with overlap-add
  - NSNet2推論 (ONNX Runtime Web)
- UIコンポーネント (App.tsx) 実装
  - 3画面構成 (home/recording/playback)
  - Original/AI Enhanced 切替
  - WAVダウンロード
- 初回コミット完了

## 今後の改善候補

- [ ] Service Worker でオフライン対応
- [ ] IndexedDB で録音履歴保存
- [ ] Web Share API でシェア機能
- [ ] WebGPU バックエンドの優先使用
- [ ] より洗練された波形表示 (ヒストリー付き)
- [ ] PWA化 (manifest.json, アイコン)
