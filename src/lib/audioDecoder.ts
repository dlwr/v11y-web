const SAMPLE_RATE = 48000;

/**
 * Decode an audio file and convert to 48kHz mono Float32Array
 */
export async function decodeAudioFile(file: File): Promise<{
  audioData: Float32Array;
  duration: number;
}> {
  const arrayBuffer = await file.arrayBuffer();
  const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Resample to 48kHz and convert to mono
    const resampled = await resampleToMono(audioBuffer, SAMPLE_RATE);

    return {
      audioData: resampled,
      duration: resampled.length / SAMPLE_RATE,
    };
  } finally {
    await audioContext.close();
  }
}

/**
 * Resample audio to target sample rate and convert to mono
 */
async function resampleToMono(
  audioBuffer: AudioBuffer,
  targetSampleRate: number
): Promise<Float32Array> {
  const numChannels = audioBuffer.numberOfChannels;
  const originalSampleRate = audioBuffer.sampleRate;
  const originalLength = audioBuffer.length;

  // Calculate output length
  const outputLength = Math.round(
    (originalLength * targetSampleRate) / originalSampleRate
  );

  // Create offline context for resampling
  const offlineCtx = new OfflineAudioContext(
    1, // mono output
    outputLength,
    targetSampleRate
  );

  // Create buffer source
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;

  // If stereo or more, mix down to mono using channel merger/splitter
  if (numChannels > 1) {
    const merger = offlineCtx.createChannelMerger(1);
    const splitter = offlineCtx.createChannelSplitter(numChannels);

    source.connect(splitter);

    // Mix all channels with equal weight
    const gain = 1 / numChannels;
    for (let i = 0; i < numChannels; i++) {
      const gainNode = offlineCtx.createGain();
      gainNode.gain.value = gain;
      splitter.connect(gainNode, i);
      gainNode.connect(merger, 0, 0);
    }

    merger.connect(offlineCtx.destination);
  } else {
    source.connect(offlineCtx.destination);
  }

  source.start(0);

  const renderedBuffer = await offlineCtx.startRendering();
  return renderedBuffer.getChannelData(0);
}

/**
 * Supported audio MIME types for input
 */
export const SUPPORTED_AUDIO_TYPES = [
  'audio/mpeg', // MP3
  'audio/wav',
  'audio/wave',
  'audio/x-wav',
  'audio/mp4', // M4A/AAC
  'audio/aac',
  'audio/ogg',
  'audio/flac',
  'audio/webm',
];

/**
 * Accept string for file input
 */
export const AUDIO_ACCEPT = '.mp3,.wav,.m4a,.aac,.ogg,.flac,.webm,audio/*';
