import { useState, useRef, useCallback } from 'react';

export type RecordingState = 'idle' | 'recording' | 'paused' | 'processing';

interface UseRecorderReturn {
  state: RecordingState;
  duration: number;
  amplitude: number;
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

  const updateAmplitude = useCallback(() => {
    if (analyserRef.current && state === 'recording') {
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      analyserRef.current.getByteTimeDomainData(dataArray);

      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const value = (dataArray[i] - 128) / 128;
        sum += value * value;
      }
      const rms = Math.sqrt(sum / dataArray.length);
      setAmplitude(Math.min(1, rms * 3));

      const elapsed = (Date.now() - startTimeRef.current) / 1000 + pausedDurationRef.current;
      setDuration(elapsed);

      animationFrameRef.current = requestAnimationFrame(updateAmplitude);
    }
  }, [state]);

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

      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(audioContext.destination);

      processor.onaudioprocess = (e) => {
        if (state === 'recording' || chunksRef.current.length === 0) {
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
      animationFrameRef.current = requestAnimationFrame(updateAmplitude);
    } catch (error) {
      console.error('Failed to start recording:', error);
      throw error;
    }
  }, [state, updateAmplitude]);

  const stopRecording = useCallback(() => {
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
      cancelAnimationFrame(animationFrameRef.current);
      setState('paused');
    }
  }, [state, duration]);

  const resumeRecording = useCallback(() => {
    if (state === 'paused') {
      startTimeRef.current = Date.now();
      setState('recording');
      animationFrameRef.current = requestAnimationFrame(updateAmplitude);
    }
  }, [state, updateAmplitude]);

  return {
    state,
    duration,
    amplitude,
    audioData,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
  };
}
