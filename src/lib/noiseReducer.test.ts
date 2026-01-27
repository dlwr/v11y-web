import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// MockWorkerクラス - Workerの振る舞いをシミュレート
class MockWorker {
  private messageListeners: ((e: MessageEvent) => void)[] = [];
  private errorListeners: ((e: ErrorEvent) => void)[] = [];

  addEventListener(type: string, listener: (e: Event) => void) {
    if (type === 'message') {
      this.messageListeners.push(listener as (e: MessageEvent) => void);
    } else if (type === 'error') {
      this.errorListeners.push(listener as (e: ErrorEvent) => void);
    }
  }

  removeEventListener(type: string, listener: (e: Event) => void) {
    if (type === 'message') {
      this.messageListeners = this.messageListeners.filter((l) => l !== listener);
    } else if (type === 'error') {
      this.errorListeners = this.errorListeners.filter((l) => l !== listener);
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  postMessage(data: unknown, transfer?: Transferable[]) {
    // Mock implementation - 何もしない
  }

  terminate() {
    // Mock implementation
  }

  // テスト用ヘルパー: メッセージを送信
  simulateMessage(data: unknown) {
    const event = { data } as MessageEvent;
    this.messageListeners.forEach((l) => l(event));
  }

  // テスト用ヘルパー: エラーを発火
  simulateError(message: string) {
    const event = { message } as ErrorEvent;
    this.errorListeners.forEach((l) => l(event));
  }

  // テスト用ヘルパー: エラーリスナーの数を取得
  getErrorListenerCount() {
    return this.errorListeners.length;
  }
}

// グローバルWorkerをモック
let mockWorkerInstance: MockWorker | null = null;

class WorkerMock extends MockWorker {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  constructor(url: URL | string, options?: WorkerOptions) {
    super();
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    mockWorkerInstance = this;
  }
}

vi.stubGlobal('Worker', WorkerMock);

describe('noiseReducer', () => {
  beforeEach(() => {
    vi.resetModules();
    mockWorkerInstance = null;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('initNoiseReducer', () => {
    it('should initialize worker and resolve on init-complete', async () => {
      const { initNoiseReducer } = await import('./noiseReducer');

      const initPromise = initNoiseReducer();

      // Workerがメッセージを送信
      setTimeout(() => {
        mockWorkerInstance?.simulateMessage({ type: 'init-complete' });
      }, 0);

      await expect(initPromise).resolves.toBeUndefined();
    });

    it('should reject on error message during init', async () => {
      const { initNoiseReducer } = await import('./noiseReducer');

      const initPromise = initNoiseReducer();

      setTimeout(() => {
        mockWorkerInstance?.simulateMessage({ type: 'error', error: 'Init failed' });
      }, 0);

      await expect(initPromise).rejects.toThrow('Init failed');
    });

    it('should reject on worker error event during init', async () => {
      const { initNoiseReducer } = await import('./noiseReducer');

      const initPromise = initNoiseReducer();

      setTimeout(() => {
        mockWorkerInstance?.simulateError('Worker crashed during init');
      }, 0);

      await expect(initPromise).rejects.toThrow('Worker crashed during init');
    });
  });

  describe('processAudio', () => {
    it('should resolve when worker sends process-complete', async () => {
      const { initNoiseReducer, processAudio } = await import('./noiseReducer');

      // まず初期化
      const initPromise = initNoiseReducer();
      setTimeout(() => {
        mockWorkerInstance?.simulateMessage({ type: 'init-complete' });
      }, 0);
      await initPromise;

      // 音声処理
      const audioData = new Float32Array([0.1, 0.2, 0.3]);
      const processPromise = processAudio(audioData);

      const resultData = new Float32Array([0.15, 0.25, 0.35]);
      setTimeout(() => {
        mockWorkerInstance?.simulateMessage({
          type: 'process-complete',
          audioData: resultData,
        });
      }, 0);

      const result = await processPromise;
      expect(result).toEqual(resultData);
    });

    it('should reject when worker sends error message', async () => {
      const { initNoiseReducer, processAudio } = await import('./noiseReducer');

      const initPromise = initNoiseReducer();
      setTimeout(() => {
        mockWorkerInstance?.simulateMessage({ type: 'init-complete' });
      }, 0);
      await initPromise;

      const audioData = new Float32Array([0.1, 0.2, 0.3]);
      const processPromise = processAudio(audioData);

      setTimeout(() => {
        mockWorkerInstance?.simulateMessage({
          type: 'error',
          error: 'Processing failed',
        });
      }, 0);

      await expect(processPromise).rejects.toThrow('Processing failed');
    });

    it('should reject when worker fires error event', async () => {
      // このテストは現在失敗するはず（errorイベントがハンドリングされていないため）
      const { initNoiseReducer, processAudio } = await import('./noiseReducer');

      const initPromise = initNoiseReducer();
      setTimeout(() => {
        mockWorkerInstance?.simulateMessage({ type: 'init-complete' });
      }, 0);
      await initPromise;

      const audioData = new Float32Array([0.1, 0.2, 0.3]);
      const processPromise = processAudio(audioData);

      setTimeout(() => {
        mockWorkerInstance?.simulateError('Worker crashed during processing');
      }, 0);

      // 現在の実装ではerrorイベントがハンドリングされていないため、
      // このテストは失敗する（タイムアウトするか、Promiseが解決されない）
      await expect(processPromise).rejects.toThrow('Worker crashed during processing');
    });

    it('should add error event listener during processing', async () => {
      // processAudio中にerrorイベントリスナーが追加されることを確認
      const { initNoiseReducer, processAudio } = await import('./noiseReducer');

      const initPromise = initNoiseReducer();
      setTimeout(() => {
        mockWorkerInstance?.simulateMessage({ type: 'init-complete' });
      }, 0);
      await initPromise;

      const audioData = new Float32Array([0.1, 0.2, 0.3]);
      processAudio(audioData);

      // 小さなdelayを入れてイベントリスナーが追加されるのを待つ
      await new Promise((resolve) => setTimeout(resolve, 10));

      // errorリスナーが追加されているか確認
      // 現在の実装では追加されていないため、このテストは失敗する
      expect(mockWorkerInstance?.getErrorListenerCount()).toBeGreaterThan(0);
    });
  });
});
