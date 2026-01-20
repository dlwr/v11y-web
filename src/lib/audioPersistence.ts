const DB_NAME = 'v11y-audio-db';
const DB_VERSION = 1;
const STORE_NAME = 'audio-state';
const STATE_KEY = 'current-session';
const MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24 hours

export interface PersistedAudioState {
  originalAudio: Float32Array;
  processedAudio: Float32Array | null;
  duration: number;
  timestamp: number;
}

interface SerializedAudioState {
  originalAudio: number[];
  processedAudio: number[] | null;
  duration: number;
  timestamp: number;
}

let dbInstance: IDBDatabase | null = null;

function openDB(): Promise<IDBDatabase> {
  if (dbInstance) return Promise.resolve(dbInstance);

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      dbInstance = request.result;
      resolve(dbInstance);
    };
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
  });
}

function withStore<T>(
  mode: IDBTransactionMode,
  callback: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return openDB().then(
    (db) =>
      new Promise((resolve, reject) => {
        const transaction = db.transaction(STORE_NAME, mode);
        const store = transaction.objectStore(STORE_NAME);
        const request = callback(store);
        transaction.oncomplete = () => resolve(request.result);
        transaction.onerror = () => reject(transaction.error);
      }),
  );
}

function serialize(state: PersistedAudioState): SerializedAudioState {
  return {
    originalAudio: Array.from(state.originalAudio),
    processedAudio: state.processedAudio ? Array.from(state.processedAudio) : null,
    duration: state.duration,
    timestamp: state.timestamp,
  };
}

function deserialize(data: SerializedAudioState): PersistedAudioState {
  return {
    originalAudio: new Float32Array(data.originalAudio),
    processedAudio: data.processedAudio ? new Float32Array(data.processedAudio) : null,
    duration: data.duration,
    timestamp: data.timestamp,
  };
}

export function saveAudioState(state: PersistedAudioState): Promise<void> {
  return withStore('readwrite', (store) => store.put(serialize(state), STATE_KEY)).then(() => {});
}

export async function loadAudioState(): Promise<PersistedAudioState | null> {
  try {
    const data = await withStore<SerializedAudioState | undefined>('readonly', (store) =>
      store.get(STATE_KEY),
    );
    return data ? deserialize(data) : null;
  } catch {
    return null;
  }
}

export async function clearAudioState(): Promise<void> {
  try {
    await withStore('readwrite', (store) => store.delete(STATE_KEY));
  } catch {
    // Ignore errors when clearing
  }
}

export function isStateValid(state: PersistedAudioState): boolean {
  return Date.now() - state.timestamp < MAX_AGE_MS;
}
