import { Mp3Encoder } from '@breezystack/lamejs';

const SAMPLE_RATE = 48000;
const BIT_RATE = 192; // kbps
const CHUNK_SIZE = 1152; // MP3 frame size

/**
 * Convert Float32Array to MP3 Blob
 */
export function floatToMp3(audioData: Float32Array): Blob {
  // Convert Float32 to Int16
  const int16Data = floatToInt16(audioData);

  // Create encoder (mono, 48kHz, 192kbps)
  const encoder = new Mp3Encoder(1, SAMPLE_RATE, BIT_RATE);

  const mp3Chunks: ArrayBuffer[] = [];

  // Encode in chunks
  for (let i = 0; i < int16Data.length; i += CHUNK_SIZE) {
    const chunk = int16Data.subarray(i, i + CHUNK_SIZE);
    const mp3Chunk = encoder.encodeBuffer(chunk);
    if (mp3Chunk.length > 0) {
      mp3Chunks.push(new Uint8Array(mp3Chunk).buffer);
    }
  }

  // Flush remaining data
  const finalChunk = encoder.flush();
  if (finalChunk.length > 0) {
    mp3Chunks.push(new Uint8Array(finalChunk).buffer);
  }

  return new Blob(mp3Chunks, { type: 'audio/mpeg' });
}

/**
 * Convert Float32Array (-1 to 1) to Int16Array
 */
function floatToInt16(floatData: Float32Array): Int16Array {
  const int16Data = new Int16Array(floatData.length);

  for (let i = 0; i < floatData.length; i++) {
    const sample = Math.max(-1, Math.min(1, floatData[i]));
    int16Data[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }

  return int16Data;
}
