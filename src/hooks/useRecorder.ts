import { useState, useRef, useCallback } from 'react';

export type RecordingState = 'idle' | 'recording' | 'paused' | 'processing';

interface UseRecorderReturn {
  state: RecordingState;
  duration: number;
  amplitude: number;
  frequencyData: number[];
  audioData: Float32Array | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  pauseRecording: () => void;
  resumeRecording: () => void;
}

const SAMPLE_RATE = 48000;

export function useRecorder(): UseRecorderReturn {
  const [state, setState] = useState<RecordingState>('idle');
  const [duration, setDuration] = useState(0);
  const [amplitude, setAmplitude] = useState(0);
  const [frequencyData, setFrequencyData] = useState<number[]>([]);
  const [audioData, setAudioData] = useState<Float32Array | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const startTimeRef = useRef<number>(0);
  const pausedDurationRef = useRef<number>(0);
  const animationFrameRef = useRef<number>(0);
  const isCapturingRef = useRef<boolean>(false);
  const isAnimatingRef = useRef<boolean>(false);

  // Animation loop ref - allows self-referencing without ESLint errors
  const animationLoopRef = useRef<(() => void) | undefined>(undefined);

  const startAnimationLoop = useCallback(() => {
    const loop = () => {
      if (!isAnimatingRef.current) return;

      if (analyserRef.current) {
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteTimeDomainData(dataArray);

        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const value = (dataArray[i] - 128) / 128;
          sum += value * value;
        }
        const rms = Math.sqrt(sum / dataArray.length);
        // Amplify the signal more (was *3, now *30) for better visualization
        const amplifiedRms = Math.min(1, rms * 30);
        setAmplitude(amplifiedRms);

        // Get frequency data for waveform visualization (sample 20 bars)
        const freqArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(freqArray);

        // Use logarithmic frequency scaling for more balanced visualization
        // Human hearing is logarithmic, so lower frequencies have more energy
        const binWidth = SAMPLE_RATE / analyserRef.current.fftSize;
        const minFreq = 100;
        const maxFreq = 4000;

        const bars: number[] = [];
        for (let i = 0; i < 20; i++) {
          // Logarithmic frequency mapping
          const t = i / 19;
          const freq = minFreq * Math.pow(maxFreq / minFreq, t);
          const binIndex = Math.floor(freq / binWidth);

          // Stronger compensation for low frequencies (they naturally have more energy)
          const freqCompensation = 0.3 + (i / 19) * 2.5; // 0.3 to 2.8
          const normalizedValue = Math.min(1, (freqArray[binIndex] / 255) * freqCompensation);
          bars.push(normalizedValue);
        }
        setFrequencyData(bars);

        const elapsed = (Date.now() - startTimeRef.current) / 1000 + pausedDurationRef.current;
        setDuration(elapsed);
      }

      // Schedule next frame using ref
      animationFrameRef.current = requestAnimationFrame(() => animationLoopRef.current?.());
    };

    animationLoopRef.current = loop;
    animationFrameRef.current = requestAnimationFrame(loop);
  }, [setAmplitude, setDuration]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;

      // Use ScriptProcessorNode for raw audio data capture
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      // Create a silent gain node to connect analyser without audio output
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;

      // Connect source to analyser (for visualization) and processor (for recording)
      source.connect(analyser);
      source.connect(processor);
      // Connect analyser through silent gain to destination (needed for analyser to work)
      analyser.connect(silentGain);
      silentGain.connect(audioContext.destination);
      // Processor needs to connect to destination to work
      processor.connect(audioContext.destination);

      isCapturingRef.current = true;
      processor.onaudioprocess = (e) => {
        if (isCapturingRef.current) {
          const inputData = e.inputBuffer.getChannelData(0);
          chunksRef.current.push(new Float32Array(inputData));
        }
      };

      audioContextRef.current = audioContext;
      streamRef.current = stream;
      sourceRef.current = source;
      processorRef.current = processor;
      analyserRef.current = analyser;
      chunksRef.current = [];
      startTimeRef.current = Date.now();
      pausedDurationRef.current = 0;

      setState('recording');
      isAnimatingRef.current = true;
      startAnimationLoop();
    } catch (error) {
      console.error('Failed to start recording:', error);
      throw error;
    }
  }, [startAnimationLoop]);

  const stopRecording = useCallback(() => {
    isAnimatingRef.current = false;
    cancelAnimationFrame(animationFrameRef.current);

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Merge all chunks
    const totalLength = chunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
    const mergedData = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunksRef.current) {
      mergedData.set(chunk, offset);
      offset += chunk.length;
    }

    setAudioData(mergedData);
    setState('idle');
    setAmplitude(0);
  }, []);

  const pauseRecording = useCallback(() => {
    if (state === 'recording') {
      pausedDurationRef.current = duration;
      isAnimatingRef.current = false;
      cancelAnimationFrame(animationFrameRef.current);
      isCapturingRef.current = false;
      setState('paused');
    }
  }, [state, duration]);

  const resumeRecording = useCallback(() => {
    if (state === 'paused') {
      startTimeRef.current = Date.now();
      isCapturingRef.current = true;
      isAnimatingRef.current = true;
      setState('recording');
      startAnimationLoop();
    }
  }, [state, startAnimationLoop]);

  return {
    state,
    duration,
    amplitude,
    frequencyData,
    audioData,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
  };
}
