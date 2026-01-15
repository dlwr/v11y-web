// Professional audio enhancement pipeline for voice
// Processes: NoiseGate -> DeEsser -> MultibandCompressor -> VoiceEQ -> Limiter

const SAMPLE_RATE = 48000;

// ============================================
// Noise Gate - Silence the silent parts
// ============================================
interface NoiseGateParams {
  threshold: number; // dB below which audio is gated
  attack: number; // ms
  release: number; // ms
  holdTime: number; // ms to hold gate open after signal drops
}

function applyNoiseGate(
  audio: Float32Array,
  params: NoiseGateParams = {
    threshold: -45,
    attack: 1,
    release: 50,
    holdTime: 100,
  }
): Float32Array {
  const output = new Float32Array(audio.length);
  const attackSamples = Math.floor((params.attack / 1000) * SAMPLE_RATE);
  const releaseSamples = Math.floor((params.release / 1000) * SAMPLE_RATE);
  const holdSamples = Math.floor((params.holdTime / 1000) * SAMPLE_RATE);
  const thresholdLinear = Math.pow(10, params.threshold / 20);

  // RMS window for level detection
  const windowSize = Math.floor(SAMPLE_RATE * 0.01); // 10ms window
  let gateOpen = false;
  let holdCounter = 0;
  let envelope = 0;

  for (let i = 0; i < audio.length; i++) {
    // Calculate RMS level
    let sumSquared = 0;
    const start = Math.max(0, i - windowSize);
    const len = i - start;
    if (len > 0) {
      for (let j = start; j < i; j++) {
        sumSquared += audio[j] * audio[j];
      }
      const rms = Math.sqrt(sumSquared / len);

      // Gate logic
      if (rms > thresholdLinear) {
        gateOpen = true;
        holdCounter = holdSamples;
      } else if (holdCounter > 0) {
        holdCounter--;
      } else {
        gateOpen = false;
      }
    }

    // Envelope follower for smooth transitions
    const targetEnvelope = gateOpen ? 1 : 0;
    const attackCoeff = 1 - Math.exp(-1 / attackSamples);
    const releaseCoeff = 1 - Math.exp(-1 / releaseSamples);
    const coeff = targetEnvelope > envelope ? attackCoeff : releaseCoeff;
    envelope += coeff * (targetEnvelope - envelope);

    output[i] = audio[i] * envelope;
  }

  return output;
}

// ============================================
// De-Esser - Reduce sibilance (s, sh, etc.)
// ============================================
interface DeEsserParams {
  frequency: number; // Center frequency for sibilance detection (Hz)
  threshold: number; // dB threshold for triggering
  ratio: number; // Compression ratio when triggered
  range: number; // Maximum reduction in dB
}

function applyDeEsser(
  audio: Float32Array,
  params: DeEsserParams = {
    frequency: 6000,
    threshold: -20,
    ratio: 4,
    range: 8,
  }
): Float32Array {
  const output = new Float32Array(audio.length);

  // Bandpass filter coefficients for sibilance band (4-8 kHz)
  const lowCut = 4000;
  const highCut = 9000;

  // Create bandpass filtered version for detection
  const detected = applyBandpassFilter(audio, lowCut, highCut);

  // Apply dynamic reduction based on detected sibilance
  const thresholdLinear = Math.pow(10, params.threshold / 20);
  const maxReduction = Math.pow(10, -params.range / 20);

  // Envelope for sibilance level
  const attackSamples = Math.floor(0.001 * SAMPLE_RATE); // 1ms attack
  const releaseSamples = Math.floor(0.05 * SAMPLE_RATE); // 50ms release
  let envelope = 0;

  for (let i = 0; i < audio.length; i++) {
    const detectedLevel = Math.abs(detected[i]);

    // Envelope follower
    const coeff = detectedLevel > envelope
      ? 1 - Math.exp(-1 / attackSamples)
      : 1 - Math.exp(-1 / releaseSamples);
    envelope += coeff * (detectedLevel - envelope);

    // Calculate gain reduction
    let gain = 1;
    if (envelope > thresholdLinear) {
      const overThreshold = envelope / thresholdLinear;
      const targetReduction = Math.pow(overThreshold, 1 - 1 / params.ratio);
      gain = 1 / targetReduction;
      gain = Math.max(gain, maxReduction);
    }

    // Apply gain reduction only to the high frequencies
    const sibilanceRatio = Math.abs(detected[i]) / (Math.abs(audio[i]) + 1e-10);
    const effectiveGain = 1 - sibilanceRatio * (1 - gain);
    output[i] = audio[i] * Math.max(0.5, effectiveGain);
  }

  return output;
}

// ============================================
// Multiband Compressor - Professional voice dynamics
// ============================================
interface BandParams {
  lowFreq: number;
  highFreq: number;
  threshold: number; // dB
  ratio: number;
  attack: number; // ms
  release: number; // ms
  makeupGain: number; // dB
}

interface MultibandCompressorParams {
  bands: BandParams[];
}

function applyMultibandCompressor(
  audio: Float32Array,
  params: MultibandCompressorParams = {
    bands: [
      // Low band - warmth and body
      { lowFreq: 20, highFreq: 200, threshold: -24, ratio: 3, attack: 10, release: 100, makeupGain: 2 },
      // Low-mid band - fullness
      { lowFreq: 200, highFreq: 800, threshold: -20, ratio: 2.5, attack: 8, release: 80, makeupGain: 1 },
      // Mid band - presence and clarity
      { lowFreq: 800, highFreq: 3000, threshold: -18, ratio: 2, attack: 5, release: 60, makeupGain: 2 },
      // High-mid band - articulation
      { lowFreq: 3000, highFreq: 8000, threshold: -22, ratio: 2, attack: 3, release: 50, makeupGain: 1 },
      // High band - air
      { lowFreq: 8000, highFreq: 20000, threshold: -26, ratio: 1.5, attack: 2, release: 40, makeupGain: 0 },
    ],
  }
): Float32Array {
  const output = new Float32Array(audio.length);

  // Process each band and sum
  for (const band of params.bands) {
    // Split band
    const bandAudio = applyBandpassFilter(audio, band.lowFreq, band.highFreq);

    // Compress band
    const compressed = compressBand(bandAudio, band);

    // Add to output
    for (let i = 0; i < output.length; i++) {
      output[i] += compressed[i];
    }
  }

  return output;
}

function compressBand(audio: Float32Array, band: BandParams): Float32Array {
  const output = new Float32Array(audio.length);
  const attackSamples = Math.floor((band.attack / 1000) * SAMPLE_RATE);
  const releaseSamples = Math.floor((band.release / 1000) * SAMPLE_RATE);
  const thresholdLinear = Math.pow(10, band.threshold / 20);
  const makeupLinear = Math.pow(10, band.makeupGain / 20);

  let envelope = 0;

  for (let i = 0; i < audio.length; i++) {
    const inputLevel = Math.abs(audio[i]);

    // Envelope follower
    const coeff = inputLevel > envelope
      ? 1 - Math.exp(-1 / attackSamples)
      : 1 - Math.exp(-1 / releaseSamples);
    envelope += coeff * (inputLevel - envelope);

    // Calculate gain
    let gain = 1;
    if (envelope > thresholdLinear) {
      const overThreshold = envelope / thresholdLinear;
      gain = Math.pow(overThreshold, 1 / band.ratio - 1);
    }

    output[i] = audio[i] * gain * makeupLinear;
  }

  return output;
}

// ============================================
// Voice EQ - Optimize frequency response for voice
// ============================================
interface VoiceEQParams {
  highPassFreq: number;
  lowShelfFreq: number;
  lowShelfGain: number; // dB
  presenceFreq: number;
  presenceGain: number; // dB
  presenceQ: number;
  highShelfFreq: number;
  highShelfGain: number; // dB
}

function applyVoiceEQ(
  audio: Float32Array,
  params: VoiceEQParams = {
    highPassFreq: 80,
    lowShelfFreq: 200,
    lowShelfGain: 1.5,
    presenceFreq: 3500,
    presenceGain: 2.5,
    presenceQ: 1.2,
    highShelfFreq: 10000,
    highShelfGain: 2,
  }
): Float32Array {
  let processed = audio;

  processed = applyHighPassFilter(processed, params.highPassFreq);
  processed = applyLowShelfFilter(processed, params.lowShelfFreq, params.lowShelfGain);
  processed = applyPeakFilter(processed, params.presenceFreq, params.presenceGain, params.presenceQ);
  processed = applyHighShelfFilter(processed, params.highShelfFreq, params.highShelfGain);

  return processed;
}

// ============================================
// Filter implementations
// ============================================

function applyHighPassFilter(input: Float32Array, cutoff: number): Float32Array {
  const output = new Float32Array(input.length);
  const omega = (2 * Math.PI * cutoff) / SAMPLE_RATE;
  const alpha = Math.sin(omega) / (2 * 0.707);

  const b0 = (1 + Math.cos(omega)) / 2;
  const b1 = -(1 + Math.cos(omega));
  const b2 = (1 + Math.cos(omega)) / 2;
  const a0 = 1 + alpha;
  const a1 = -2 * Math.cos(omega);
  const a2 = 1 - alpha;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
    output[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }

  return output;
}

function applyBandpassFilter(input: Float32Array, lowCut: number, highCut: number): Float32Array {
  let processed = applyHighPassFilter(input, lowCut);
  processed = applyLowPassFilter(processed, highCut);
  return processed;
}

function applyLowPassFilter(input: Float32Array, cutoff: number): Float32Array {
  const output = new Float32Array(input.length);
  const omega = (2 * Math.PI * cutoff) / SAMPLE_RATE;
  const alpha = Math.sin(omega) / (2 * 0.707);

  const b0 = (1 - Math.cos(omega)) / 2;
  const b1 = 1 - Math.cos(omega);
  const b2 = (1 - Math.cos(omega)) / 2;
  const a0 = 1 + alpha;
  const a1 = -2 * Math.cos(omega);
  const a2 = 1 - alpha;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
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
  const sinOmega = Math.sin(omega);
  const cosOmega = Math.cos(omega);
  const alpha = (sinOmega / 2) * Math.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2);

  const b0 = A * ((A + 1) - (A - 1) * cosOmega + 2 * Math.sqrt(A) * alpha);
  const b1 = 2 * A * ((A - 1) - (A + 1) * cosOmega);
  const b2 = A * ((A + 1) - (A - 1) * cosOmega - 2 * Math.sqrt(A) * alpha);
  const a0 = (A + 1) + (A - 1) * cosOmega + 2 * Math.sqrt(A) * alpha;
  const a1 = -2 * ((A - 1) + (A + 1) * cosOmega);
  const a2 = (A + 1) + (A - 1) * cosOmega - 2 * Math.sqrt(A) * alpha;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
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
  const sinOmega = Math.sin(omega);
  const cosOmega = Math.cos(omega);
  const alpha = (sinOmega / 2) * Math.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2);

  const b0 = A * ((A + 1) + (A - 1) * cosOmega + 2 * Math.sqrt(A) * alpha);
  const b1 = -2 * A * ((A - 1) + (A + 1) * cosOmega);
  const b2 = A * ((A + 1) + (A - 1) * cosOmega - 2 * Math.sqrt(A) * alpha);
  const a0 = (A + 1) - (A - 1) * cosOmega + 2 * Math.sqrt(A) * alpha;
  const a1 = 2 * ((A - 1) - (A + 1) * cosOmega);
  const a2 = (A + 1) - (A - 1) * cosOmega - 2 * Math.sqrt(A) * alpha;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
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
  const sinOmega = Math.sin(omega);
  const cosOmega = Math.cos(omega);
  const alpha = sinOmega / (2 * Q);

  const b0 = 1 + alpha * A;
  const b1 = -2 * cosOmega;
  const b2 = 1 - alpha * A;
  const a0 = 1 + alpha / A;
  const a1 = -2 * cosOmega;
  const a2 = 1 - alpha / A;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
    output[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }

  return output;
}

// ============================================
// Brick-wall Limiter - Prevent clipping
// ============================================
function applyLimiter(audio: Float32Array, ceiling: number = -0.3): Float32Array {
  const output = new Float32Array(audio.length);
  const ceilingLinear = Math.pow(10, ceiling / 20);

  const lookaheadSamples = Math.floor(SAMPLE_RATE * 0.005); // 5ms lookahead
  const releaseSamples = Math.floor(SAMPLE_RATE * 0.1); // 100ms release
  const releaseCoeff = 1 - Math.exp(-1 / releaseSamples);

  let gainReduction = 1;

  for (let i = 0; i < audio.length; i++) {
    let peak = 0;
    for (let j = 0; j < lookaheadSamples && i + j < audio.length; j++) {
      peak = Math.max(peak, Math.abs(audio[i + j]));
    }

    const targetGain = peak > ceilingLinear ? ceilingLinear / peak : 1;

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
export interface EnhancementOptions {
  noiseGate?: boolean;
  deEsser?: boolean;
  multibandCompressor?: boolean;
  voiceEQ?: boolean;
  limiter?: boolean;
}

export function enhanceVoice(
  audio: Float32Array,
  options: EnhancementOptions = {
    noiseGate: true,
    deEsser: true,
    multibandCompressor: true,
    voiceEQ: true,
    limiter: true,
  }
): Float32Array {
  let processed = audio;

  // 1. Noise Gate - Clean up silent parts first
  if (options.noiseGate) {
    processed = applyNoiseGate(processed);
  }

  // 2. De-Esser - Reduce harsh sibilance before compression
  if (options.deEsser) {
    processed = applyDeEsser(processed);
  }

  // 3. Multiband Compressor - Even out dynamics
  if (options.multibandCompressor) {
    processed = applyMultibandCompressor(processed);
  }

  // 4. Voice EQ - Shape the frequency response
  if (options.voiceEQ) {
    processed = applyVoiceEQ(processed);
  }

  // 5. Limiter - Prevent clipping
  if (options.limiter) {
    processed = applyLimiter(processed);
  }

  return processed;
}
