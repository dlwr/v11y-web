import { useState, useRef, useEffect, useCallback } from 'react';
import { useRecorder } from './hooks/useRecorder';
import { initNoiseReducer, processAudio } from './lib/noiseReducer';
import { floatToWav, formatDuration, downloadBlob } from './lib/audioUtils';
import './App.css';

type AppState = 'home' | 'recording' | 'playback';

// Keep screen awake during recording
function useWakeLock(enabled: boolean) {
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);

  useEffect(() => {
    const requestWakeLock = async () => {
      if (enabled && 'wakeLock' in navigator) {
        try {
          wakeLockRef.current = await navigator.wakeLock.request('screen');
        } catch (err) {
          console.warn('Wake Lock not available:', err);
        }
      }
    };

    const releaseWakeLock = () => {
      if (wakeLockRef.current) {
        wakeLockRef.current.release();
        wakeLockRef.current = null;
      }
    };

    if (enabled) {
      requestWakeLock();
    } else {
      releaseWakeLock();
    }

    return () => releaseWakeLock();
  }, [enabled]);
}

function App() {
  const [appState, setAppState] = useState<AppState>('home');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedAudio, setProcessedAudio] = useState<Float32Array | null>(null);
  const [useProcessed, setUseProcessed] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [modelLoading, setModelLoading] = useState(true);

  // Keep screen awake during recording
  useWakeLock(appState === 'recording');

  const audioRef = useRef<HTMLAudioElement>(null);
  const audioUrlRef = useRef<string | null>(null);

  const {
    state: recordingState,
    duration,
    frequencyData,
    audioData,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
  } = useRecorder();

  // Preload model
  useEffect(() => {
    initNoiseReducer()
      .then(() => setModelLoading(false))
      .catch((err) => {
        console.error('Failed to load model:', err);
        setModelLoading(false);
      });
  }, []);

  const processRecording = useCallback(async (data: Float32Array) => {
    setIsProcessing(true);
    try {
      const processed = await processAudio(data);
      setProcessedAudio(processed);
    } catch (error) {
      console.error('Failed to process audio:', error);
      setProcessedAudio(null);
    }
    setIsProcessing(false);
  }, []);

  // Process audio when new audioData becomes available
  const prevAudioDataRef = useRef<Float32Array | null>(null);
  useEffect(() => {
    if (audioData && audioData !== prevAudioDataRef.current && audioData.length > 0) {
      prevAudioDataRef.current = audioData;
      // Use queueMicrotask to avoid synchronous setState in effect
      queueMicrotask(() => {
        processRecording(audioData);
      });
    }
  }, [audioData, processRecording]);

  // Update audio URL when source changes
  useEffect(() => {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }

    const data = useProcessed && processedAudio ? processedAudio : audioData;
    if (data) {
      const blob = floatToWav(data);
      audioUrlRef.current = URL.createObjectURL(blob);
      if (audioRef.current) {
        audioRef.current.src = audioUrlRef.current;
      }
    }

    return () => {
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, [useProcessed, processedAudio, audioData]);

  const handleStartRecording = async () => {
    setAppState('recording');
    await startRecording();
  };

  const handleStopRecording = () => {
    stopRecording();
    // State will transition to playback and start processing
    // We need to wait for audioData to be available
    setTimeout(() => {
      setAppState('playback');
    }, 0);
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setPlaybackTime(audioRef.current.currentTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setPlaybackTime(0);
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleDownload = () => {
    const data = useProcessed && processedAudio ? processedAudio : audioData;
    if (data) {
      const blob = floatToWav(data);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      downloadBlob(blob, `recording-${timestamp}.wav`);
    }
  };

  const handleNewRecording = () => {
    setAppState('home');
    setProcessedAudio(null);
    setPlaybackTime(0);
    setIsPlaying(false);
  };

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      setPlaybackTime(time);
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 text-white">
      <audio
        ref={audioRef}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        onPause={handlePause}
        className="hidden"
      />

      {/* Header */}
      <h1 className="text-3xl font-bold mb-2">v11y</h1>
      <p className="text-gray-400 mb-8 text-sm">AI Noise Reduction Recorder</p>

      {modelLoading && (
        <div className="fixed top-4 right-4 bg-blue-600 px-4 py-2 rounded-lg text-sm">
          Loading AI model...
        </div>
      )}

      {/* Home Screen */}
      {appState === 'home' && (
        <div className="flex flex-col items-center">
          <button
            onClick={handleStartRecording}
            className="w-24 h-24 rounded-full bg-red-500 hover:bg-red-600 transition-colors flex items-center justify-center shadow-lg"
          >
            <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="8" />
            </svg>
          </button>
          <p className="mt-4 text-gray-400">Tap to record</p>
        </div>
      )}

      {/* Recording Screen */}
      {appState === 'recording' && (
        <div className="flex flex-col items-center w-full max-w-md">
          {/* Waveform visualization - frequency bars */}
          <div className="w-full h-32 bg-gray-800/50 rounded-lg mb-6 flex items-end justify-center gap-1 overflow-hidden px-4">
            {(frequencyData.length > 0 ? frequencyData : Array(20).fill(0)).map((value, i) => {
              const barHeight = Math.max(8, value * 100);
              return (
                <div
                  key={i}
                  className="flex-1 bg-red-500 rounded-t transition-all duration-75"
                  style={{ height: `${barHeight}%` }}
                />
              );
            })}
          </div>

          {/* Duration */}
          <p className="text-4xl font-mono mb-8">{formatDuration(duration)}</p>

          {/* Controls */}
          <div className="flex items-center gap-6">
            {recordingState === 'recording' && (
              <button
                onClick={pauseRecording}
                className="w-14 h-14 rounded-full bg-gray-700 hover:bg-gray-600 flex items-center justify-center"
              >
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              </button>
            )}

            {recordingState === 'paused' && (
              <button
                onClick={resumeRecording}
                className="w-14 h-14 rounded-full bg-gray-700 hover:bg-gray-600 flex items-center justify-center"
              >
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <polygon points="5,3 19,12 5,21" />
                </svg>
              </button>
            )}

            <button
              onClick={handleStopRecording}
              className="w-20 h-20 rounded-full bg-red-500 hover:bg-red-600 flex items-center justify-center"
            >
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            </button>
          </div>

          <p className="mt-4 text-gray-400 text-sm">
            {recordingState === 'paused' ? 'Paused' : 'Recording...'}
          </p>
        </div>
      )}

      {/* Playback Screen */}
      {appState === 'playback' && (
        <div className="flex flex-col items-center w-full max-w-md">
          {isProcessing && (
            <div className="mb-6 flex items-center gap-2 text-blue-400">
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              <span>Processing with AI...</span>
            </div>
          )}

          {/* Toggle */}
          <div className="flex items-center gap-4 mb-6">
            <button
              onClick={() => setUseProcessed(false)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                !useProcessed ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              Original
            </button>
            <button
              onClick={() => setUseProcessed(true)}
              disabled={!processedAudio}
              className={`px-4 py-2 rounded-lg transition-colors ${
                useProcessed ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
              } ${!processedAudio ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              AI Enhanced
            </button>
          </div>

          {/* Playback time */}
          <p className="text-4xl font-mono mb-4">
            {formatDuration(playbackTime)} / {formatDuration(duration)}
          </p>

          {/* Seek bar */}
          <input
            type="range"
            min={0}
            max={duration || 1}
            step={0.1}
            value={playbackTime}
            onChange={handleSeek}
            className="w-full mb-6 accent-blue-500"
          />

          {/* Controls */}
          <div className="flex items-center gap-6 mb-8">
            <button
              onClick={handlePlayPause}
              className="w-16 h-16 rounded-full bg-blue-600 hover:bg-blue-700 flex items-center justify-center"
            >
              {isPlaying ? (
                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                  <polygon points="5,3 19,12 5,21" />
                </svg>
              )}
            </button>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleDownload}
              className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                />
              </svg>
              Download
            </button>

            <button
              onClick={handleNewRecording}
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg"
            >
              New Recording
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
