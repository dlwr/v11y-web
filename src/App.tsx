import { useState, useRef, useEffect, useCallback } from 'react';

import { useRecorder } from './hooks/useRecorder';
import { decodeAudioFile, AUDIO_ACCEPT } from './lib/audioDecoder';
import {
  saveAudioState,
  loadAudioState,
  clearAudioState,
  isStateValid,
} from './lib/audioPersistence';
import { floatToWav, formatDuration, downloadBlob } from './lib/audioUtils';
import { floatToMp3 } from './lib/mp3Encoder';
import { initNoiseReducer, processAudio } from './lib/noiseReducer';
import './App.css';

type AppState = 'home' | 'recording' | 'playback';
type OutputFormat = 'wav' | 'mp3';

// Request notification permission and send notification
async function sendNotification(title: string, body: string): Promise<void> {
  if (!('Notification' in window)) return;

  if (Notification.permission === 'default') {
    await Notification.requestPermission();
  }

  if (Notification.permission === 'granted') {
    new Notification(title, {
      body,
      icon: '/pwa-192x192.png',
      badge: '/pwa-192x192.png',
      tag: 'v11y-processing',
    });
  }
}

// Keep screen awake during long-running operations
function useWakeLock(enabled: boolean): void {
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
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('wav');
  const [uploadedAudio, setUploadedAudio] = useState<Float32Array | null>(null);
  const [uploadedDuration, setUploadedDuration] = useState(0);
  const [isDecoding, setIsDecoding] = useState(false);
  const [isEncoding, setIsEncoding] = useState(false);

  // Keep screen awake during recording and any processing
  const shouldKeepAwake = appState === 'recording' || isProcessing || isDecoding || isEncoding;
  useWakeLock(shouldKeepAwake);

  const audioRef = useRef<HTMLAudioElement>(null);
  const audioUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  // Track restored state for re-processing
  const restoredStateRef = useRef<{ audio: Float32Array; duration: number } | null>(null);
  // Track processing state for avoiding duplicate processing
  const isProcessingRef = useRef(false);

  // Preload model and restore saved state
  useEffect(() => {
    const init = async () => {
      // Try to restore saved state first
      try {
        const savedState = await loadAudioState();
        if (savedState && isStateValid(savedState)) {
          setUploadedAudio(savedState.originalAudio);
          setUploadedDuration(savedState.duration);
          setProcessedAudio(savedState.processedAudio);
          setAppState('playback');
          // If processing was interrupted (no processed audio), queue for re-processing
          if (!savedState.processedAudio) {
            restoredStateRef.current = {
              audio: savedState.originalAudio,
              duration: savedState.duration,
            };
          }
        }
      } catch (err) {
        console.warn('Failed to restore audio state:', err);
      }

      // Load AI model
      try {
        await initNoiseReducer();
        setModelLoading(false);
      } catch (err) {
        console.error('Failed to load model:', err);
        setModelLoading(false);
      }
    };
    init();
  }, []);

  const processRecording = useCallback(async (data: Float32Array, audioDuration: number) => {
    // Prevent duplicate processing
    if (isProcessingRef.current) {
      console.warn('processRecording called while already processing, ignoring');
      return;
    }
    isProcessingRef.current = true;
    setIsProcessing(true);
    // Save original audio immediately before processing starts
    // This way if the app sleeps during processing, we can restore and retry
    await saveAudioState({
      originalAudio: data,
      processedAudio: null,
      duration: audioDuration,
      timestamp: Date.now(),
    });
    try {
      const processed = await processAudio(data);
      setProcessedAudio(processed);
      // Update with processed audio
      await saveAudioState({
        originalAudio: data,
        processedAudio: processed,
        duration: audioDuration,
        timestamp: Date.now(),
      });
      // Send notification when processing is complete (useful when app is in background)
      if (document.hidden) {
        sendNotification('v11y', 'AI noise reduction complete! Your recording is ready.');
      }
    } catch (error) {
      console.error('Failed to process audio:', error);
      setProcessedAudio(null);
      if (document.hidden) {
        sendNotification('v11y', 'Processing failed. Please try again.');
      }
    }
    isProcessingRef.current = false;
    setIsProcessing(false);
  }, []);

  // Re-process restored audio once model is loaded
  useEffect(() => {
    if (!modelLoading && restoredStateRef.current) {
      const { audio, duration: audioDuration } = restoredStateRef.current;
      restoredStateRef.current = null;
      processRecording(audio, audioDuration);
    }
  }, [modelLoading, processRecording]);

  // Process audio when new audioData becomes available
  const prevAudioDataRef = useRef<Float32Array | null>(null);
  useEffect(() => {
    if (audioData && audioData !== prevAudioDataRef.current && audioData.length > 0) {
      prevAudioDataRef.current = audioData;
      // Use queueMicrotask to avoid synchronous setState in effect
      queueMicrotask(() => {
        processRecording(audioData, duration);
      });
    }
  }, [audioData, duration, processRecording]);

  // Update audio URL when source changes
  useEffect(() => {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }

    const currentAudioData = uploadedAudio || audioData;
    const data = useProcessed && processedAudio ? processedAudio : currentAudioData;
    if (data) {
      const blob = floatToWav(data);
      audioUrlRef.current = URL.createObjectURL(blob);
      if (audioRef.current) {
        audioRef.current.src = audioUrlRef.current;
        // Reset playback state when source changes (use queueMicrotask to avoid setState in effect)
        queueMicrotask(() => {
          setIsPlaying(false);
          setPlaybackTime(0);
        });
      }
    }

    return () => {
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, [useProcessed, processedAudio, audioData, uploadedAudio]);

  const handleStartRecording = async () => {
    // Request notification permission on first recording
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
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

  const handleDownload = async () => {
    const isEnhanced = useProcessed && processedAudio;
    const currentAudioData = uploadedAudio || audioData;
    const data = isEnhanced ? processedAudio : currentAudioData;
    if (data) {
      let blob: Blob;
      if (outputFormat === 'mp3') {
        setIsEncoding(true);
        try {
          blob = await floatToMp3(data);
        } finally {
          setIsEncoding(false);
        }
      } else {
        blob = floatToWav(data);
      }
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const suffix = isEnhanced ? 'enhanced' : 'original';
      downloadBlob(blob, `recording-${timestamp}-${suffix}.${outputFormat}`);
    }
  };

  const handleNewRecording = () => {
    setAppState('home');
    setProcessedAudio(null);
    setUploadedAudio(null);
    setUploadedDuration(0);
    setPlaybackTime(0);
    setIsPlaying(false);
    // Clear persisted state
    clearAudioState();
  };

  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsDecoding(true);
    try {
      const { audioData: decodedAudio, duration: decodedDuration } = await decodeAudioFile(file);
      setUploadedAudio(decodedAudio);
      setUploadedDuration(decodedDuration);
      setAppState('playback');
      // Process the uploaded audio
      processRecording(decodedAudio, decodedDuration);
    } catch (error) {
      console.error('Failed to decode audio file:', error);
      alert('Failed to decode audio file. Please try another file.');
    } finally {
      setIsDecoding(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
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
      <p className="text-gray-400 mb-1 text-sm">AI Noise Reduction Recorder</p>
      <p className="text-gray-600 mb-8 text-xs">v{__APP_VERSION__}</p>

      {modelLoading && (
        <div className="fixed top-4 right-4 bg-blue-600 px-4 py-2 rounded-lg text-sm">
          Loading AI model...
        </div>
      )}

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={AUDIO_ACCEPT}
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Home Screen */}
      {appState === 'home' && (
        <div className="flex flex-col items-center">
          <button
            onClick={handleStartRecording}
            disabled={isDecoding}
            className="w-24 h-24 rounded-full bg-red-500 hover:bg-red-600 transition-colors flex items-center justify-center shadow-lg disabled:opacity-50"
          >
            <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="8" />
            </svg>
          </button>
          <p className="mt-4 text-gray-400">Tap to record</p>

          {/* Divider */}
          <div className="flex items-center gap-4 my-6 w-full max-w-xs">
            <div className="flex-1 h-px bg-gray-600" />
            <span className="text-gray-500 text-sm">or</span>
            <div className="flex-1 h-px bg-gray-600" />
          </div>

          {/* Upload button */}
          <button
            onClick={handleFileSelect}
            disabled={isDecoding}
            className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50"
          >
            {isDecoding ? (
              <>
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
                <span>Decoding...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                  />
                </svg>
                <span>Upload Audio File</span>
              </>
            )}
          </button>
          <p className="mt-2 text-gray-500 text-xs">MP3, WAV, M4A</p>
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
            {formatDuration(playbackTime)} / {formatDuration(uploadedDuration || duration)}
          </p>

          {/* Seek bar */}
          <input
            type="range"
            min={0}
            max={(uploadedDuration || duration) || 1}
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

          {/* Output format */}
          <div className="flex items-center gap-2 mb-6">
            <span className="text-gray-400 text-sm mr-2">Format:</span>
            <button
              onClick={() => setOutputFormat('wav')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                outputFormat === 'wav' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              WAV
            </button>
            <button
              onClick={() => setOutputFormat('mp3')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                outputFormat === 'mp3' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              MP3
            </button>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleDownload}
              disabled={isEncoding}
              className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isEncoding ? (
                <>
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
                  <span>Encoding...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                  </svg>
                  <span>Download {outputFormat.toUpperCase()}</span>
                </>
              )}
            </button>

            <button
              onClick={handleNewRecording}
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg"
            >
              New
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
