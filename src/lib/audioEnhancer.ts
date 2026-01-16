// Professional audio enhancement pipeline for voice
// Simple but effective: HighPass -> Compressor -> VoiceEQ -> Limiter

const SAMPLE_RATE = 48000;

// ============================================
// Simple Compressor - Make voice consistent
// ============================================
function applyCompressor(
  audio: Float32Array,
  threshold: number = -20, // dB
  ratio: number = 4,
  attack: number = 5, // ms
  release: number = 100, // ms
  makeupGain: number = 6 // dB
): Float32Array {
  const output = new Float32Array(audio.length);
  const attackSamples = Math.floor((attack / 1000) * SAMPLE_RATE);
  const releaseSamples = Math.floor((release / 1000) * SAMPLE_RATE);
  const thresholdLinear = Math.pow(10, threshold / 20);
  const makeupLinear = Math.pow(10, makeupGain / 20);

  let envelope = 0;

  for (let i = 0; i < audio.length; i++) {
    const inputLevel = Math.abs(audio[i]);

    // Envelope follower
    const coeff = inputLevel > envelope
      ? 1 - Math.exp(-1 / attackSamples)
      : 1 - Math.exp(-1 / releaseSamples);
    envelope += coeff * (inputLevel - envelope);

    // Calculate gain reduction
    let gain = 1;
    if (envelope > thresholdLinear) {
      const overDb = 20 * Math.log10(envelope / thresholdLinear);
      const reducedDb = overDb * (1 - 1 / ratio);
      gain = Math.pow(10, -reducedDb / 20);
    }

    output[i] = audio[i] * gain * makeupLinear;
  }

  return output;
}

// ============================================
// Voice EQ - Shape the frequency response
// ============================================
function applyVoiceEQ(audio: Float32Array): Float32Array {
  let processed = audio;

  // 1. High-pass at 80Hz - Remove rumble
  processed = applyHighPassFilter(processed, 80);

  // 2. Low shelf boost at 200Hz (+3dB) - Add warmth
  processed = applyLowShelfFilter(processed, 200, 3);

  // 3. Presence peak at 3kHz (+4dB, Q=1.0) - Clarity
  processed = applyPeakFilter(processed, 3000, 4, 1.0);

  // 4. High shelf boost at 8kHz (+3dB) - Air/brightness
  processed = applyHighShelfFilter(processed, 8000, 3);

  return processed;
}

// ============================================
// Biquad Filter implementations (IIR)
// ============================================

function applyHighPassFilter(input: Float32Array, cutoff: number): Float32Array {
  const output = new Float32Array(input.length);
  const omega = (2 * Math.PI * cutoff) / SAMPLE_RATE;
  const cosOmega = Math.cos(omega);
  const alpha = Math.sin(omega) / (2 * 0.707);

  const b0 = (1 + cosOmega) / 2;
  const b1 = -(1 + cosOmega);
  const b2 = (1 + cosOmega) / 2;
  const a0 = 1 + alpha;
  const a1 = -2 * cosOmega;
  const a2 = 1 - alpha;

  // Normalize coefficients
  const nb0 = b0 / a0;
  const nb1 = b1 / a0;
  const nb2 = b2 / a0;
  const na1 = a1 / a0;
  const na2 = a2 / a0;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2;
    output[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }

  return output;
}

function applyLowShelfFilter(input: Float32Array, frequency: number, gainDb: number): Float32Array {
  const output = new Float32Array(input.length);
  const A = Math.pow(10, gainDb / 40);
  const omega = (2 * Math.PI * frequency) / SAMPLE_RATE;
  const cosOmega = Math.cos(omega);
  const sinOmega = Math.sin(omega);
  const sqrtA = Math.sqrt(A);
  const alpha = (sinOmega / 2) * Math.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2);

  const b0 = A * ((A + 1) - (A - 1) * cosOmega + 2 * sqrtA * alpha);
  const b1 = 2 * A * ((A - 1) - (A + 1) * cosOmega);
  const b2 = A * ((A + 1) - (A - 1) * cosOmega - 2 * sqrtA * alpha);
  const a0 = (A + 1) + (A - 1) * cosOmega + 2 * sqrtA * alpha;
  const a1 = -2 * ((A - 1) + (A + 1) * cosOmega);
  const a2 = (A + 1) + (A - 1) * cosOmega - 2 * sqrtA * alpha;

  const nb0 = b0 / a0;
  const nb1 = b1 / a0;
  const nb2 = b2 / a0;
  const na1 = a1 / a0;
  const na2 = a2 / a0;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2;
    output[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }

  return output;
}

function applyHighShelfFilter(input: Float32Array, frequency: number, gainDb: number): Float32Array {
  const output = new Float32Array(input.length);
  const A = Math.pow(10, gainDb / 40);
  const omega = (2 * Math.PI * frequency) / SAMPLE_RATE;
  const cosOmega = Math.cos(omega);
  const sinOmega = Math.sin(omega);
  const sqrtA = Math.sqrt(A);
  const alpha = (sinOmega / 2) * Math.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2);

  const b0 = A * ((A + 1) + (A - 1) * cosOmega + 2 * sqrtA * alpha);
  const b1 = -2 * A * ((A - 1) + (A + 1) * cosOmega);
  const b2 = A * ((A + 1) + (A - 1) * cosOmega - 2 * sqrtA * alpha);
  const a0 = (A + 1) - (A - 1) * cosOmega + 2 * sqrtA * alpha;
  const a1 = 2 * ((A - 1) - (A + 1) * cosOmega);
  const a2 = (A + 1) - (A - 1) * cosOmega - 2 * sqrtA * alpha;

  const nb0 = b0 / a0;
  const nb1 = b1 / a0;
  const nb2 = b2 / a0;
  const na1 = a1 / a0;
  const na2 = a2 / a0;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2;
    output[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }

  return output;
}

function applyPeakFilter(input: Float32Array, frequency: number, gainDb: number, Q: number): Float32Array {
  const output = new Float32Array(input.length);
  const A = Math.pow(10, gainDb / 40);
  const omega = (2 * Math.PI * frequency) / SAMPLE_RATE;
  const cosOmega = Math.cos(omega);
  const sinOmega = Math.sin(omega);
  const alpha = sinOmega / (2 * Q);

  const b0 = 1 + alpha * A;
  const b1 = -2 * cosOmega;
  const b2 = 1 - alpha * A;
  const a0 = 1 + alpha / A;
  const a1 = -2 * cosOmega;
  const a2 = 1 - alpha / A;

  const nb0 = b0 / a0;
  const nb1 = b1 / a0;
  const nb2 = b2 / a0;
  const na1 = a1 / a0;
  const na2 = a2 / a0;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2;
    output[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }

  return output;
}

// ============================================
// Brick-wall Limiter - Prevent clipping
// ============================================
function applyLimiter(audio: Float32Array, ceiling: number = -1): Float32Array {
  const output = new Float32Array(audio.length);
  const ceilingLinear = Math.pow(10, ceiling / 20);

  const lookaheadSamples = Math.floor(SAMPLE_RATE * 0.005); // 5ms
  const releaseSamples = Math.floor(SAMPLE_RATE * 0.05); // 50ms
  const releaseCoeff = 1 - Math.exp(-1 / releaseSamples);

  let gainReduction = 1;

  for (let i = 0; i < audio.length; i++) {
    // Find peak in lookahead window
    let peak = Math.abs(audio[i]);
    for (let j = 1; j < lookaheadSamples && i + j < audio.length; j++) {
      peak = Math.max(peak, Math.abs(audio[i + j]));
    }

    const targetGain = peak > ceilingLinear ? ceilingLinear / peak : 1;

    // Instant attack, slow release
    if (targetGain < gainReduction) {
      gainReduction = targetGain;
    } else {
      gainReduction += releaseCoeff * (targetGain - gainReduction);
    }

    output[i] = audio[i] * gainReduction;
  }

  return output;
}

// ============================================
// Main Enhancement Pipeline
// ============================================
export function enhanceVoice(audio: Float32Array): Float32Array {
  // Simple, effective chain:
  // 1. Compressor - Even out dynamics, make voice punchy
  let processed = applyCompressor(audio);

  // 2. Voice EQ - Shape frequency response
  processed = applyVoiceEQ(processed);

  // 3. Limiter - Prevent clipping
  processed = applyLimiter(processed);

  return processed;
}
