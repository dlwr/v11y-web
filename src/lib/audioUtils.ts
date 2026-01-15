const SAMPLE_RATE = 48000;

// Target loudness for podcast: -16 LUFS
const TARGET_LUFS = -16;

/**
 * Calculate the loudness of audio in LUFS (Loudness Units Full Scale)
 * Simplified ITU-R BS.1770 implementation for mono audio
 */
function calculateLoudness(audioData: Float32Array): number {
  if (audioData.length === 0) return -Infinity;

  // K-weighting filter coefficients for 48kHz
  // High shelf filter stage 1
  const highShelfB = [1.53512485958697, -2.69169618940638, 1.19839281085285];
  const highShelfA = [1, -1.69065929318241, 0.73248077421585];

  // High pass filter stage 2
  const highPassB = [1.0, -2.0, 1.0];
  const highPassA = [1, -1.99004745483398, 0.99007225036621];

  // Apply K-weighting filters
  let filtered = applyBiquadFilter(audioData, highShelfB, highShelfA);
  filtered = applyBiquadFilter(filtered, highPassB, highPassA);

  // Calculate mean square
  let sumSquared = 0;
  for (let i = 0; i < filtered.length; i++) {
    sumSquared += filtered[i] * filtered[i];
  }
  const meanSquare = sumSquared / filtered.length;

  // Convert to LUFS
  if (meanSquare === 0) return -Infinity;
  return -0.691 + 10 * Math.log10(meanSquare);
}

/**
 * Apply a biquad filter to audio data
 */
function applyBiquadFilter(
  input: Float32Array,
  b: number[],
  a: number[]
): Float32Array {
  const output = new Float32Array(input.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = (b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2) / a[0];
    output[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }

  return output;
}

/**
 * Normalize audio to target LUFS (-16 LUFS for podcast)
 * Returns normalized audio data
 */
export function normalizeLoudness(
  audioData: Float32Array,
  targetLufs: number = TARGET_LUFS
): Float32Array {
  const currentLufs = calculateLoudness(audioData);

  if (!isFinite(currentLufs)) {
    return audioData;
  }

  // Calculate gain needed to reach target
  const gainDb = targetLufs - currentLufs;
  const gain = Math.pow(10, gainDb / 20);

  // Apply gain with limiter to prevent clipping
  const output = new Float32Array(audioData.length);
  let peakAfterGain = 0;

  for (let i = 0; i < audioData.length; i++) {
    const sample = audioData[i] * gain;
    peakAfterGain = Math.max(peakAfterGain, Math.abs(sample));
    output[i] = sample;
  }

  // If peak exceeds 1.0, apply limiting
  if (peakAfterGain > 0.99) {
    const limitGain = 0.99 / peakAfterGain;
    for (let i = 0; i < output.length; i++) {
      output[i] *= limitGain;
    }
  }

  return output;
}

export function floatToWav(audioData: Float32Array): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = SAMPLE_RATE * blockAlign;
  const dataSize = audioData.length * bytesPerSample;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write audio data
  let offset = 44;
  for (let i = 0; i < audioData.length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(offset, intSample, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
