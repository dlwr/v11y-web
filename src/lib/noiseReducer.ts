// Web Worker wrapper for noise reduction
let worker: Worker | null = null;
let initPromise: Promise<void> | null = null;

export async function initNoiseReducer(): Promise<void> {
  if (worker) return;

  if (initPromise) {
    return initPromise;
  }

  initPromise = new Promise((resolve, reject) => {
    worker = new Worker(new URL('./noiseReducer.worker.ts', import.meta.url), {
      type: 'module',
    });

    const handleMessage = (e: MessageEvent) => {
      if (e.data.type === 'init-complete') {
        worker?.removeEventListener('message', handleMessage);
        resolve();
      } else if (e.data.type === 'error') {
        worker?.removeEventListener('message', handleMessage);
        reject(new Error(e.data.error));
      }
    };

    const handleError = (e: ErrorEvent) => {
      worker?.removeEventListener('message', handleMessage);
      worker?.removeEventListener('error', handleError);
      console.error('Worker error:', e);
      reject(new Error(e.message || 'Worker initialization failed'));
    };

    worker.addEventListener('message', handleMessage);
    worker.addEventListener('error', handleError);
    worker.postMessage({ type: 'init' });
  });

  return initPromise;
}

export async function processAudio(audioData: Float32Array): Promise<Float32Array> {
  if (!worker) {
    await initNoiseReducer();
  }

  if (!worker) {
    throw new Error('Failed to initialize NoiseReducer Worker');
  }

  return new Promise((resolve, reject) => {
    const handleMessage = (e: MessageEvent) => {
      if (e.data.type === 'process-complete') {
        worker?.removeEventListener('message', handleMessage);
        resolve(e.data.audioData);
      } else if (e.data.type === 'error') {
        worker?.removeEventListener('message', handleMessage);
        reject(new Error(e.data.error));
      }
    };

    worker!.addEventListener('message', handleMessage);
    // Transfer the buffer for better performance
    const audioDataCopy = new Float32Array(audioData);
    worker!.postMessage({ type: 'process', audioData: audioDataCopy }, [audioDataCopy.buffer]);
  });
}

export function releaseNoiseReducer(): void {
  if (worker) {
    worker.terminate();
    worker = null;
    initPromise = null;
  }
}
