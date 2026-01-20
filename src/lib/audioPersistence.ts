const DB_NAME = 'v11y-audio-db';
const DB_VERSION = 1;
const STORE_NAME = 'audio-state';
const STATE_KEY = 'current-session';

export interface PersistedAudioState {
  originalAudio: Float32Array;
  processedAudio: Float32Array | null;
  duration: number;
  timestamp: number;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
  });
}

export async function saveAudioState(state: PersistedAudioState): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    // Convert Float32Arrays to regular arrays for storage
    const serializable = {
      originalAudio: Array.from(state.originalAudio),
      processedAudio: state.processedAudio ? Array.from(state.processedAudio) : null,
      duration: state.duration,
      timestamp: state.timestamp,
    };

    const request = store.put(serializable, STATE_KEY);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

export async function loadAudioState(): Promise<PersistedAudioState | null> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(STATE_KEY);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const data = request.result;
        if (!data) {
          resolve(null);
          return;
        }

        // Convert arrays back to Float32Arrays
        const state: PersistedAudioState = {
          originalAudio: new Float32Array(data.originalAudio),
          processedAudio: data.processedAudio ? new Float32Array(data.processedAudio) : null,
          duration: data.duration,
          timestamp: data.timestamp,
        };
        resolve(state);
      };
    });
  } catch {
    return null;
  }
}

export async function clearAudioState(): Promise<void> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(STATE_KEY);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  } catch {
    // Ignore errors when clearing
  }
}

// Check if saved state is still valid (within 24 hours)
export function isStateValid(state: PersistedAudioState): boolean {
  const MAX_AGE = 24 * 60 * 60 * 1000; // 24 hours
  return Date.now() - state.timestamp < MAX_AGE;
}
