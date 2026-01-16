// Professional audio enhancement pipeline for voice
// Chain: NoiseGate -> ClickRemover -> BreathReducer -> DeEsser -> Compressor -> VoiceEQ -> Limiter

const SAMPLE_RATE = 48000;

// ============================================
// Noise Gate - Silence quiet parts completely
// ============================================
function applyNoiseGate(
  audio: Float32Array,
  threshold: number = -40, // dB - gate opens above this
  holdTime: number = 100, // ms - keep gate open after signal drops
  attack: number = 1, // ms - how fast gate opens
  release: number = 50 // ms - how fast gate closes
): Float32Array {
  const output = new Float32Array(audio.length);
  const thresholdLinear = Math.pow(10, threshold / 20);
  const holdSamples = Math.floor((holdTime / 1000) * SAMPLE_RATE);
  const attackCoeff = 1 - Math.exp(-1 / Math.floor((attack / 1000) * SAMPLE_RATE));
  const releaseCoeff = 1 - Math.exp(-1 / Math.floor((release / 1000) * SAMPLE_RATE));

  // RMS window for level detection (10ms)
  const rmsWindowSize = Math.floor(SAMPLE_RATE * 0.01);
  let gateOpen = false;
  let holdCounter = 0;
  let gateGain = 0;

  for (let i = 0; i < audio.length; i++) {
    // Calculate RMS level
    let sumSquared = 0;
    const start = Math.max(0, i - rmsWindowSize);
    for (let j = start; j <= i; j++) {
      sumSquared += audio[j] * audio[j];
    }
    const rms = Math.sqrt(sumSquared / (i - start + 1));

    // Gate logic with hold
    if (rms > thresholdLinear) {
      gateOpen = true;
      holdCounter = holdSamples;
    } else if (holdCounter > 0) {
      holdCounter--;
    } else {
      gateOpen = false;
    }

    // Smooth gate transitions
    const targetGain = gateOpen ? 1 : 0;
    const coeff = targetGain > gateGain ? attackCoeff : releaseCoeff;
    gateGain += coeff * (targetGain - gateGain);

    output[i] = audio[i] * gateGain;
  }

  return output;
}

// ============================================
// Click/Pop Remover - Remove short transient noises
// ============================================
function applyClickRemover(audio: Float32Array): Float32Array {
  const output = new Float32Array(audio);

  // Detect clicks by looking for sudden amplitude changes
  const windowSize = Math.floor(SAMPLE_RATE * 0.002); // 2ms window
  const clickThreshold = 3.0; // Ratio of sample to local average

  for (let i = windowSize; i < audio.length - windowSize; i++) {
    // Calculate local average (excluding current sample)
    let sum = 0;
    for (let j = i - windowSize; j < i; j++) {
      sum += Math.abs(audio[j]);
    }
    const localAvg = sum / windowSize;

    // If current sample is much larger than local average, it's likely a click
    if (localAvg > 0.0001 && Math.abs(audio[i]) > localAvg * clickThreshold) {
      // Check if it's a short transient (not sustained)
      let isClick = true;
      for (let j = 1; j <= Math.min(windowSize, audio.length - i - 1); j++) {
        if (Math.abs(audio[i + j]) > localAvg * clickThreshold * 0.7) {
          isClick = false; // Sustained signal, not a click
          break;
        }
      }

      if (isClick) {
        // Interpolate to remove click
        const before = audio[i - 1];
        const after = audio[Math.min(i + 2, audio.length - 1)];
        output[i] = (before + after) / 2;
        output[i + 1] = after * 0.7 + before * 0.3;
      }
    }
  }

  return output;
}

// ============================================
// Breath Reducer - Reduce breath sounds between words
// ============================================
function applyBreathReducer(
  audio: Float32Array,
  reduction: number = 12 // dB reduction for detected breaths
): Float32Array {
  const output = new Float32Array(audio.length);
  const reductionLinear = Math.pow(10, -reduction / 20);

  // Breath characteristics: low frequency content, moderate amplitude
  const frameSize = Math.floor(SAMPLE_RATE * 0.03); // 30ms frames

  // Analyze in frames
  const numFrames = Math.floor(audio.length / frameSize);
  const frameEnergy: number[] = [];
  const frameLowRatio: number[] = [];

  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSize;
    let totalEnergy = 0;
    let lowEnergy = 0;

    // Simple energy calculation
    for (let i = 0; i < frameSize; i++) {
      const sample = audio[start + i];
      totalEnergy += sample * sample;
    }

    // Low-pass filtered energy (approximate low frequency content)
    let lpState = 0;
    const lpCoeff = 0.1;
    for (let i = 0; i < frameSize; i++) {
      lpState += lpCoeff * (audio[start + i] - lpState);
      lowEnergy += lpState * lpState;
    }

    frameEnergy.push(totalEnergy / frameSize);
    frameLowRatio.push(totalEnergy > 0.00001 ? lowEnergy / totalEnergy : 0);
  }

  // Find overall speech level
  const sortedEnergy = [...frameEnergy].sort((a, b) => b - a);
  const speechLevel = sortedEnergy[Math.floor(numFrames * 0.1)] || 0.001; // Top 10% as speech

  // Detect breath frames: moderate energy, high low-frequency ratio
  const breathFrames: boolean[] = [];
  for (let f = 0; f < numFrames; f++) {
    const energyRatio = frameEnergy[f] / speechLevel;
    const isBreath = energyRatio > 0.01 && energyRatio < 0.3 && frameLowRatio[f] > 0.5;
    breathFrames.push(isBreath);
  }

  // Apply reduction with smooth transitions
  let currentGain = 1;
  const smoothCoeff = 1 - Math.exp(-1 / Math.floor(SAMPLE_RATE * 0.01));

  for (let i = 0; i < audio.length; i++) {
    const frameIdx = Math.min(Math.floor(i / frameSize), numFrames - 1);
    const targetGain = breathFrames[frameIdx] ? reductionLinear : 1;
    currentGain += smoothCoeff * (targetGain - currentGain);
    output[i] = audio[i] * currentGain;
  }

  return output;
}

// ============================================
// De-Esser - Reduce harsh sibilance (s, sh sounds)
// Aggressive settings for strong friction sound reduction
// ============================================
function applyDeEsser(
  audio: Float32Array,
  threshold: number = -30, // dB - lower threshold = trigger earlier (was -25)
  reduction: number = 15 // dB - stronger max reduction (was 8)
): Float32Array {
  const output = new Float32Array(audio.length);

  // Wider bandpass filter to detect sibilance (3-12 kHz range for more coverage)
  const detected = applyBandpassFilter(audio, 3000, 12000);

  const thresholdLinear = Math.pow(10, threshold / 20);
  const maxReduction = Math.pow(10, -reduction / 20);

  // Faster attack for quicker response to sibilants
  const attackSamples = Math.floor(0.0002 * SAMPLE_RATE); // 0.2ms (was 0.5ms)
  const releaseSamples = Math.floor(0.025 * SAMPLE_RATE); // 25ms (was 30ms)
  let envelope = 0;

  for (let i = 0; i < audio.length; i++) {
    const detectedLevel = Math.abs(detected[i]);

    // Envelope follower
    const coeff = detectedLevel > envelope
      ? 1 - Math.exp(-1 / attackSamples)
      : 1 - Math.exp(-1 / releaseSamples);
    envelope += coeff * (detectedLevel - envelope);

    // Calculate gain reduction based on sibilance level
    let gain = 1;
    if (envelope > thresholdLinear) {
      // More aggressive ratio (0.85 instead of 0.7) for harder compression
      const overDb = 20 * Math.log10(envelope / thresholdLinear);
      const reductionDb = Math.min(overDb * 0.85, reduction);
      gain = Math.pow(10, -reductionDb / 20);
      gain = Math.max(gain, maxReduction);
    }

    // Apply reduction more aggressively to sibilance content
    const sibilanceRatio = Math.min(1, Math.abs(detected[i]) / (Math.abs(audio[i]) + 1e-10));
    // Increase the effect multiplier (1.3x) for more noticeable reduction
    const effectiveGain = 1 - Math.min(1, sibilanceRatio * 1.3) * (1 - gain);
    output[i] = audio[i] * effectiveGain;
  }

  return output;
}

function applyBandpassFilter(input: Float32Array, lowCut: number, highCut: number): Float32Array {
  // High-pass then low-pass
  let processed = applyHighPassFilter(input, lowCut);
  processed = applyLowPassFilter(processed, highCut);
  return processed;
}

function applyLowPassFilter(input: Float32Array, cutoff: number): Float32Array {
  const output = new Float32Array(input.length);
  const omega = (2 * Math.PI * cutoff) / SAMPLE_RATE;
  const cosOmega = Math.cos(omega);
  const alpha = Math.sin(omega) / (2 * 0.707);

  const b0 = (1 - cosOmega) / 2;
  const b1 = 1 - cosOmega;
  const b2 = (1 - cosOmega) / 2;
  const a0 = 1 + alpha;
  const a1 = -2 * cosOmega;
  const a2 = 1 - alpha;

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
  // 1. Noise Gate - Remove quiet noise between speech
  let processed = applyNoiseGate(audio);

  // 2. Click Remover - Remove mouth clicks and pops
  processed = applyClickRemover(processed);

  // 3. Breath Reducer - Reduce breath sounds
  processed = applyBreathReducer(processed);

  // 4. De-Esser - Reduce harsh sibilance (before compression amplifies it)
  processed = applyDeEsser(processed);

  // 5. Compressor - Even out dynamics, make voice punchy
  processed = applyCompressor(processed);

  // 6. Voice EQ - Shape frequency response
  processed = applyVoiceEQ(processed);

  // 7. Limiter - Prevent clipping
  processed = applyLimiter(processed);

  return processed;
}
