// Professional audio enhancement pipeline for voice
// Chain: AdaptiveNoiseGate -> ClickRemover -> BreathReducer -> DeEsser -> Compressor -> VoiceEQ -> Limiter

const SAMPLE_RATE = 48000;

// ============================================
// Constants for Audio Analysis and Processing
// ============================================

// Frame sizes for analysis
const ANALYSIS_FRAME_MS = 30; // 30ms frames for audio analysis
const ANALYSIS_FRAME_SIZE = Math.floor(SAMPLE_RATE * (ANALYSIS_FRAME_MS / 1000));

// SNR thresholds for adaptive processing (in dB)
const SNR_VERY_HIGH = 35;
const SNR_HIGH = 25;
const SNR_MEDIUM = 15;

// Adaptive gate margin based on SNR (in dB)
const GATE_MARGIN_VERY_HIGH_SNR = 8;
const GATE_MARGIN_HIGH_SNR = 12;
const GATE_MARGIN_MEDIUM_SNR = 18;
const GATE_MARGIN_LOW_SNR = 25;

// Speech protection thresholds
const SPEECH_PROTECTION_DB = 10; // Max dB protection for high speech probability
const QUIET_SPEECH_OFFSET_DB = 22; // dB below peak for quiet speech estimate
const QUIET_SPEECH_HEADROOM_DB = 6; // Additional headroom for quiet speech

// Speech likelihood thresholds for gate protection
const SPEECH_PROB_HIGH = 0.6;
const SPEECH_PROB_MEDIUM = 0.35;
const SPEECH_PROB_LOW = 0.15;

// Minimum gain preservation based on speech probability
const MIN_GAIN_HIGH_SPEECH = 0.6;
const MIN_GAIN_MEDIUM_SPEECH = 0.35;
const MIN_GAIN_LOW_SPEECH = 0.15;

// ============================================
// Type Definitions
// ============================================

interface AudioAnalysis {
  p10Db: number; // 10th percentile (noise floor estimate)
  p90Db: number; // 90th percentile (speech peaks)
  estimatedSnr: number; // Signal-to-noise ratio
  speechLikelihoodFrames: number[]; // Per-frame speech probability
}

interface AdaptiveGateConfig {
  threshold: number; // Calculated adaptive threshold in dB
  holdTime: number; // Adjusted hold time in ms
  softKnee: number; // Soft knee width in dB
}

function analyzeAudioLevels(audio: Float32Array): AudioAnalysis {
  const numFrames = Math.floor(audio.length / ANALYSIS_FRAME_SIZE);

  if (numFrames === 0) {
    return {
      p10Db: -60,
      p90Db: -30,
      estimatedSnr: 30,
      speechLikelihoodFrames: [],
    };
  }

  // Calculate RMS for each frame
  const frameRmsDb: number[] = [];
  for (let f = 0; f < numFrames; f++) {
    const start = f * ANALYSIS_FRAME_SIZE;
    let sumSquared = 0;
    for (let i = 0; i < ANALYSIS_FRAME_SIZE; i++) {
      sumSquared += audio[start + i] * audio[start + i];
    }
    const rms = Math.sqrt(sumSquared / ANALYSIS_FRAME_SIZE);
    const rmsDb = 20 * Math.log10(rms + 1e-10);
    frameRmsDb.push(rmsDb);
  }

  // Sort for percentile calculation
  const sorted = [...frameRmsDb].sort((a, b) => a - b);
  const p10Db = sorted[Math.floor(numFrames * 0.1)] ?? -80;
  const p90Db = sorted[Math.floor(numFrames * 0.9)] ?? -30;
  const estimatedSnr = p90Db - p10Db;

  // Calculate speech likelihood for each frame
  const speechLikelihoodFrames = calculateSpeechLikelihood(
    audio,
    numFrames,
    ANALYSIS_FRAME_SIZE
  );

  return {
    p10Db,
    p90Db,
    estimatedSnr,
    speechLikelihoodFrames,
  };
}

function calculateSpeechLikelihood(
  audio: Float32Array,
  numFrames: number,
  frameSize: number
): number[] {
  const likelihood: number[] = [];

  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSize;

    // Feature 1: Energy
    let energy = 0;
    for (let i = 0; i < frameSize; i++) {
      energy += audio[start + i] * audio[start + i];
    }
    energy /= frameSize;

    // Feature 2: Zero-crossing rate (ZCR) - speech typically has lower ZCR than noise
    let zcr = 0;
    for (let i = 1; i < frameSize; i++) {
      if ((audio[start + i] >= 0) !== (audio[start + i - 1] >= 0)) {
        zcr++;
      }
    }
    zcr /= frameSize;

    // Feature 3: Low-frequency energy ratio (speech has more low-freq content)
    const cutoff = 500;
    const alpha = 1 - Math.exp((-2 * Math.PI * cutoff) / SAMPLE_RATE);
    let lpState = 0;
    let lowEnergy = 0;
    for (let i = 0; i < frameSize; i++) {
      lpState += alpha * (audio[start + i] - lpState);
      lowEnergy += lpState * lpState;
    }
    lowEnergy /= frameSize;
    const lfRatio = lowEnergy / (energy + 1e-10);

    // Feature 4: Autocorrelation at lag 1 (high = periodic = likely speech)
    let ac1 = 0;
    for (let i = 1; i < frameSize; i++) {
      ac1 += audio[start + i] * audio[start + i - 1];
    }
    ac1 /= frameSize - 1;
    const normalizedAc1 = ac1 / (energy + 1e-10);

    // Combine features into speech likelihood
    let score = 0;

    // Energy contribution (higher energy = more likely speech)
    // Adjusted to be more sensitive to quiet speech (-60dB baseline, gentler curve)
    const energyDb = 10 * Math.log10(energy + 1e-10);
    score += 0.25 * sigmoid((energyDb + 60) / 12);

    // ZCR contribution (lower ZCR = more likely speech)
    score += 0.2 * (1 - sigmoid((zcr - 0.1) / 0.05));

    // Low-frequency ratio (higher = more likely voiced speech)
    score += 0.25 * sigmoid((lfRatio - 0.3) / 0.1);

    // Autocorrelation (higher = more periodic = likely speech)
    score += 0.25 * sigmoid(normalizedAc1 / 0.3);

    likelihood.push(Math.max(0, Math.min(1, score)));
  }

  // Apply minimal median filter to preserve quick speech transitions
  // Reduced from 3 to 1 to avoid smoothing out short quiet speech segments
  return medianFilter(likelihood, 1);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function medianFilter(arr: number[], windowSize: number): number[] {
  const result: number[] = [];
  const halfWindow = Math.floor(windowSize / 2);

  for (let i = 0; i < arr.length; i++) {
    const start = Math.max(0, i - halfWindow);
    const end = Math.min(arr.length, i + halfWindow + 1);
    const window = arr.slice(start, end).sort((a, b) => a - b);
    result.push(window[Math.floor(window.length / 2)]);
  }

  return result;
}

function calculateAdaptiveThreshold(analysis: AudioAnalysis): AdaptiveGateConfig {
  const { p10Db, p90Db, estimatedSnr } = analysis;

  // Calculate margin based on SNR - more conservative to preserve quiet speech
  let margin: number;
  if (estimatedSnr > SNR_VERY_HIGH) {
    margin = GATE_MARGIN_VERY_HIGH_SNR;
  } else if (estimatedSnr > SNR_HIGH) {
    margin = GATE_MARGIN_HIGH_SNR;
  } else if (estimatedSnr > SNR_MEDIUM) {
    margin = GATE_MARGIN_MEDIUM_SNR;
  } else {
    margin = GATE_MARGIN_LOW_SNR;
  }

  // Base threshold: noise floor + margin
  let threshold = p10Db + margin;

  // Ensure threshold doesn't cut quiet speech (whispers can be 20-25dB below peak)
  const quietSpeechEstimate = p90Db - QUIET_SPEECH_OFFSET_DB;
  const maxThreshold = quietSpeechEstimate - QUIET_SPEECH_HEADROOM_DB;
  threshold = Math.min(threshold, maxThreshold);

  // Absolute limits - lowered to preserve very quiet speech
  threshold = Math.max(-70, Math.min(-30, threshold));

  // Adjust hold time and soft knee based on SNR
  const holdTime = estimatedSnr > 20 ? 100 : estimatedSnr > 10 ? 150 : 200;
  const softKnee = estimatedSnr > 20 ? 3 : 6;

  return { threshold, holdTime, softKnee };
}

// ============================================
// Adaptive Noise Gate - Preserves quiet speech
// ============================================
function applyAdaptiveNoiseGate(
  audio: Float32Array,
  config: AdaptiveGateConfig,
  speechLikelihood: number[]
): Float32Array {
  const output = new Float32Array(audio.length);

  const { threshold, holdTime, softKnee } = config;
  const thresholdLinear = Math.pow(10, threshold / 20);
  const softKneeLinear = softKnee / 20;

  const holdSamples = Math.floor((holdTime / 1000) * SAMPLE_RATE);
  const attackCoeff =
    1 - Math.exp(-1 / Math.floor((1 / 1000) * SAMPLE_RATE)); // 1ms attack
  const releaseCoeff =
    1 - Math.exp(-1 / Math.floor((50 / 1000) * SAMPLE_RATE)); // 50ms release

  const rmsWindowSize = Math.floor(SAMPLE_RATE * 0.01); // 10ms RMS window
  const frameSize = ANALYSIS_FRAME_SIZE; // Match VAD frame size

  let gateGain = 0;
  let holdCounter = 0;

  for (let i = 0; i < audio.length; i++) {
    // Calculate RMS level
    let sumSquared = 0;
    const start = Math.max(0, i - rmsWindowSize);
    for (let j = start; j <= i; j++) {
      sumSquared += audio[j] * audio[j];
    }
    const rms = Math.sqrt(sumSquared / (i - start + 1));

    // Get speech likelihood for this sample
    const frameIdx = Math.min(
      Math.floor(i / frameSize),
      speechLikelihood.length - 1
    );
    const speechProb = speechLikelihood[frameIdx] ?? 0;

    // Adjust effective threshold based on speech likelihood
    // If high speech probability, lower the threshold to preserve speech
    const speechProtection = speechProb * SPEECH_PROTECTION_DB;
    const effectiveThreshold =
      thresholdLinear * Math.pow(10, -speechProtection / 20);

    // Soft knee calculation
    let targetGain: number;
    if (rms > effectiveThreshold * (1 + softKneeLinear)) {
      // Above threshold + knee: full open
      targetGain = 1;
      holdCounter = holdSamples;
    } else if (rms > effectiveThreshold * (1 - softKneeLinear)) {
      // In soft knee region: gradual transition
      const kneePosition =
        (rms / effectiveThreshold - (1 - softKneeLinear)) / (2 * softKneeLinear);
      targetGain = kneePosition;
      if (kneePosition > 0.5) holdCounter = holdSamples;
    } else if (holdCounter > 0) {
      // In hold period
      holdCounter--;
      targetGain = 1;
    } else {
      // Below threshold: close gate
      targetGain = 0;
    }

    // Additional speech protection: never fully close if speech likely
    if (speechProb > SPEECH_PROB_HIGH) {
      targetGain = Math.max(targetGain, MIN_GAIN_HIGH_SPEECH);
    } else if (speechProb > SPEECH_PROB_MEDIUM) {
      targetGain = Math.max(targetGain, MIN_GAIN_MEDIUM_SPEECH);
    } else if (speechProb > SPEECH_PROB_LOW) {
      targetGain = Math.max(targetGain, MIN_GAIN_LOW_SPEECH);
    }

    // Smooth gate transitions
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
  // 0. Analyze audio to calculate adaptive parameters
  const analysis = analyzeAudioLevels(audio);
  const gateConfig = calculateAdaptiveThreshold(analysis);

  // 1. Adaptive Noise Gate - Remove quiet noise while preserving quiet speech
  let processed = applyAdaptiveNoiseGate(
    audio,
    gateConfig,
    analysis.speechLikelihoodFrames
  );

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
