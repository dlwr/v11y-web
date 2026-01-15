import * as ort from 'onnxruntime-web';
import { normalizeLoudness } from './audioUtils';

// NSNet2 parameters for 48kHz
const FFT_SIZE = 1024;
const NUM_BINS = FFT_SIZE / 2 + 1; // 513
const FRAME_SIZE = 960; // 20ms at 48kHz
const HOP_SIZE = 480; // 10ms, 50% overlap
const MIN_MASK = 0.15; // Minimum mask floor to preserve voice naturalness

let ortSession: ort.InferenceSession | null = null;

export async function initNoiseReducer(): Promise<void> {
  if (ortSession) return;

  console.log('Initializing ONNX Runtime...');
  ortSession = await ort.InferenceSession.create('/models/nsnet2-48k.onnx', {
    executionProviders: ['webgl', 'wasm'],
  });
  console.log('NSNet2 model loaded');
}

export async function processAudio(audioData: Float32Array): Promise<Float32Array> {
  if (!ortSession) {
    await initNoiseReducer();
  }

  if (!ortSession) {
    throw new Error('Failed to initialize NoiseReducer');
  }

  // Pad audio to ensure complete frames
  const paddedLength = Math.ceil((audioData.length - FRAME_SIZE) / HOP_SIZE + 1) * HOP_SIZE + FRAME_SIZE;
  const paddedAudio = new Float32Array(paddedLength);
  paddedAudio.set(audioData);

  const numFrames = Math.floor((paddedLength - FRAME_SIZE) / HOP_SIZE) + 1;
  console.log(`Processing ${numFrames} frames`);

  // STFT analysis
  const magnitudes: Float32Array[] = [];
  const phases: Float32Array[] = [];
  const window = createHannWindow(FRAME_SIZE);

  for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
    const startIdx = frameIdx * HOP_SIZE;
    const frame = new Float32Array(FFT_SIZE);

    // Apply window
    for (let i = 0; i < FRAME_SIZE; i++) {
      if (startIdx + i < paddedAudio.length) {
        frame[i] = paddedAudio[startIdx + i] * window[i];
      }
    }

    // Compute FFT
    const { real, imag } = computeFFT(frame);

    // Extract magnitude and phase
    const mag = new Float32Array(NUM_BINS);
    const phase = new Float32Array(NUM_BINS);

    for (let k = 0; k < NUM_BINS; k++) {
      mag[k] = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
      phase[k] = Math.atan2(imag[k], real[k]);
    }

    magnitudes.push(mag);
    phases.push(phase);
  }

  // Prepare input: log-power spectrum [1, numFrames, 513]
  const eps = 1e-8;
  const inputData = new Float32Array(numFrames * NUM_BINS);

  for (let f = 0; f < numFrames; f++) {
    for (let b = 0; b < NUM_BINS; b++) {
      const power = magnitudes[f][b] * magnitudes[f][b];
      inputData[f * NUM_BINS + b] = Math.log(power + eps);
    }
  }

  // Run inference
  console.log('Running NSNet2 inference...');
  const inputTensor = new ort.Tensor('float32', inputData, [1, numFrames, NUM_BINS]);
  const results = await ortSession.run({ input: inputTensor });
  const outputData = results[Object.keys(results)[0]].data as Float32Array;

  console.log('NSNet2 inference complete');

  // Reconstruct audio with mask applied
  const outputAudio = new Float32Array(paddedLength);
  const outputWindow = new Float32Array(paddedLength);

  for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
    const real = new Float32Array(FFT_SIZE);
    const imag = new Float32Array(FFT_SIZE);

    for (let k = 0; k < NUM_BINS; k++) {
      const mask = outputData[frameIdx * NUM_BINS + k];
      const clampedMask = Math.max(MIN_MASK, Math.min(1, mask));
      const enhancedMag = magnitudes[frameIdx][k] * clampedMask;
      const phase = phases[frameIdx][k];

      real[k] = enhancedMag * Math.cos(phase);
      imag[k] = enhancedMag * Math.sin(phase);

      // Mirror for negative frequencies
      if (k > 0 && k < NUM_BINS - 1) {
        real[FFT_SIZE - k] = real[k];
        imag[FFT_SIZE - k] = -imag[k];
      }
    }

    // Inverse FFT
    const timeFrame = computeIFFT(real, imag);

    // Overlap-add
    const startIdx = frameIdx * HOP_SIZE;
    for (let i = 0; i < FRAME_SIZE; i++) {
      if (startIdx + i < paddedLength) {
        outputAudio[startIdx + i] += timeFrame[i] * window[i];
        outputWindow[startIdx + i] += window[i] * window[i];
      }
    }
  }

  // Normalize by overlap-add window
  for (let i = 0; i < outputAudio.length; i++) {
    if (outputWindow[i] > 1e-8) {
      outputAudio[i] /= outputWindow[i];
    }
  }

  // Return original length with loudness normalization (-16 LUFS for podcast)
  const denoisedAudio = outputAudio.slice(0, audioData.length);
  console.log('Applying loudness normalization to -16 LUFS...');
  return normalizeLoudness(denoisedAudio);
}

function createHannWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let n = 0; n < size; n++) {
    window[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (size - 1)));
  }
  return window;
}

// Cooley-Tukey FFT (radix-2)
function computeFFT(input: Float32Array): { real: Float32Array; imag: Float32Array } {
  const n = input.length;
  const real = new Float32Array(input);
  const imag = new Float32Array(n);

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n / 2;
    while (k <= j) {
      j -= k;
      k /= 2;
    }
    j += k;
  }

  // Cooley-Tukey iterative FFT
  let m = 2;
  while (m <= n) {
    const wm = (-2 * Math.PI) / m;
    for (let k = 0; k < n; k += m) {
      let w = 0;
      for (let jj = 0; jj < m / 2; jj++) {
        const cosW = Math.cos(w);
        const sinW = Math.sin(w);
        const tReal = cosW * real[k + jj + m / 2] - sinW * imag[k + jj + m / 2];
        const tImag = sinW * real[k + jj + m / 2] + cosW * imag[k + jj + m / 2];
        const uReal = real[k + jj];
        const uImag = imag[k + jj];
        real[k + jj] = uReal + tReal;
        imag[k + jj] = uImag + tImag;
        real[k + jj + m / 2] = uReal - tReal;
        imag[k + jj + m / 2] = uImag - tImag;
        w += wm;
      }
    }
    m *= 2;
  }

  return { real, imag };
}

function computeIFFT(real: Float32Array, imag: Float32Array): Float32Array {
  const n = real.length;
  const inputReal = new Float32Array(real);
  const inputImag = new Float32Array(n);

  // Conjugate
  for (let i = 0; i < n; i++) {
    inputImag[i] = -imag[i];
  }

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [inputReal[i], inputReal[j]] = [inputReal[j], inputReal[i]];
      [inputImag[i], inputImag[j]] = [inputImag[j], inputImag[i]];
    }
    let k = n / 2;
    while (k <= j) {
      j -= k;
      k /= 2;
    }
    j += k;
  }

  // Cooley-Tukey iterative FFT
  let m = 2;
  while (m <= n) {
    const wm = (-2 * Math.PI) / m;
    for (let k = 0; k < n; k += m) {
      let w = 0;
      for (let jj = 0; jj < m / 2; jj++) {
        const cosW = Math.cos(w);
        const sinW = Math.sin(w);
        const tReal = cosW * inputReal[k + jj + m / 2] - sinW * inputImag[k + jj + m / 2];
        const tImag = sinW * inputReal[k + jj + m / 2] + cosW * inputImag[k + jj + m / 2];
        const uReal = inputReal[k + jj];
        const uImag = inputImag[k + jj];
        inputReal[k + jj] = uReal + tReal;
        inputImag[k + jj] = uImag + tImag;
        inputReal[k + jj + m / 2] = uReal - tReal;
        inputImag[k + jj + m / 2] = uImag - tImag;
        w += wm;
      }
    }
    m *= 2;
  }

  // Scale
  const output = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    output[i] = inputReal[i] / n;
  }
  return output;
}

export function releaseNoiseReducer(): void {
  ortSession = null;
}
