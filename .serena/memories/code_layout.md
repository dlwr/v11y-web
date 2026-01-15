# v11y-web コード構成

## 概要

ブラウザ完結型のAIノイズ除去レコーダー。v11y (Kotlin Multiplatform) のWeb版。

## 技術スタック

- **フレームワーク**: React + TypeScript + Vite
- **スタイリング**: Tailwind CSS (@tailwindcss/vite)
- **AI推論**: ONNX Runtime Web (onnxruntime-web)
- **音声処理**: WebAudio API

## ディレクトリ構造

```
v11y-web/
├── public/
│   └── models/
│       └── nsnet2-48k.onnx    # NSNet2ノイズ除去モデル (24MB)
├── src/
│   ├── App.tsx                # メインアプリ (3画面: home/recording/playback)
│   ├── App.css                # (空、Tailwindを使用)
│   ├── index.css              # Tailwindインポート + グローバルスタイル
│   ├── main.tsx               # エントリーポイント
│   ├── hooks/
│   │   └── useRecorder.ts     # 録音カスタムフック
│   └── lib/
│       ├── noiseReducer.ts    # NSNet2ノイズ除去 (STFT/ISTFT/FFT実装含む)
│       └── audioUtils.ts      # WAV変換、ダウンロード、時間フォーマット
├── vite.config.ts             # Vite設定 (React + Tailwind)
├── tsconfig.json              # TypeScript設定
└── package.json               # 依存関係
```

## 主要ファイル詳細

### src/hooks/useRecorder.ts
- WebAudio API による48kHz/mono録音
- ScriptProcessorNode でPCMデータ取得
- AnalyserNode でリアルタイム振幅取得
- 状態: `idle` | `recording` | `paused` | `processing`

### src/lib/noiseReducer.ts
- NSNet2モデル (ONNX) によるノイズ除去
- STFT: FFT size 1024, 513 bins, hop 480 samples
- Hann窓、overlap-add再構成
- 最小マスク値 0.15 (音声の自然さ保持)

### src/lib/audioUtils.ts
- `floatToWav()`: Float32Array → WAV Blob変換
- `downloadBlob()`: ファイルダウンロード
- `formatDuration()`: 秒 → MM:SS形式

### src/App.tsx
3つの画面状態を管理:
1. **home**: 録音開始ボタン
2. **recording**: 振幅表示、一時停止/再開、停止
3. **playback**: Original/AI Enhanced切替、シーク、ダウンロード

## 音声処理フロー

```
録音 (WebAudio API, 48kHz)
    ↓
Float32Array (PCMデータ)
    ↓
STFT (1024-point FFT, Hann窓)
    ↓
log-power spectrum [1, frames, 513]
    ↓
NSNet2推論 (ONNX Runtime Web)
    ↓
マスク出力 [1, frames, 513]
    ↓
マスク適用 (min 0.15) + 位相復元
    ↓
ISTFT (overlap-add)
    ↓
Float32Array (処理済み)
    ↓
WAV変換 → ダウンロード
```

## 依存関係

```json
{
  "dependencies": {
    "onnxruntime-web": "^1.x",
    "react": "^19.x",
    "react-dom": "^19.x"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.x",
    "tailwindcss": "^4.x",
    "typescript": "~5.x",
    "vite": "^7.x"
  }
}
```

## 元プロジェクトとの対応

| v11y (KMP) | v11y-web |
|------------|----------|
| AudioRecorder.android.kt | useRecorder.ts |
| NoiseReducer.android.kt | noiseReducer.ts |
| RecordingScreen.kt | App.tsx (recording state) |
| PlaybackScreen.kt | App.tsx (playback state) |
