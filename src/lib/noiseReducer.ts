// Web Worker wrapper for noise reduction
let worker: Worker | null = null;
let initPromise: Promise<void> | null = null;
let processingPromise: Promise<Float32Array> | null = null;

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
  // 既に処理中の場合はエラーを投げる（呼び出し元で排他制御すべき）
  if (processingPromise) {
    throw new Error('processAudio called while already processing');
  }

  if (!worker) {
    await initNoiseReducer();
  }

  if (!worker) {
    throw new Error('Failed to initialize NoiseReducer Worker');
  }

  const TIMEOUT_MS = 10 * 60 * 1000; // 10分タイムアウト

  processingPromise = new Promise((resolve, reject) => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const cleanup = () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      worker?.removeEventListener('message', handleMessage);
      worker?.removeEventListener('error', handleError);
    };

    const handleMessage = (e: MessageEvent) => {
      if (e.data.type === 'process-complete') {
        cleanup();
        resolve(e.data.audioData);
      } else if (e.data.type === 'error') {
        cleanup();
        reject(new Error(e.data.error));
      }
    };

    const handleError = (e: ErrorEvent) => {
      cleanup();
      reject(new Error(e.message || 'Worker processing failed'));
    };

    timeoutId = setTimeout(() => {
      cleanup();
      reject(new Error(`Processing timeout after ${TIMEOUT_MS / 1000}s`));
    }, TIMEOUT_MS);

    worker!.addEventListener('message', handleMessage);
    worker!.addEventListener('error', handleError);

    // Transfer the buffer for better performance
    const audioDataCopy = new Float32Array(audioData);
    worker!.postMessage({ type: 'process', audioData: audioDataCopy }, [audioDataCopy.buffer]);
  });

  return processingPromise.finally(() => {
    processingPromise = null;
  });
}

export function releaseNoiseReducer(): void {
  if (worker) {
    worker.terminate();
    worker = null;
    initPromise = null;
  }
}
