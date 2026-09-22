/**
 * Connector for any OpenAI-compatible chat completions endpoint.
 *
 * This covers the UVU AI Gateway, OpenAI itself, Azure OpenAI, Together,
 * Groq, OpenRouter, Ollama and LM Studio. The key is held only in this
 * browser and is sent only to the base URL the user typed in; nothing is
 * proxied through any other server.
 */

export interface LMConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  temperature: number;
  maxTokens: number;
  /** Some gateways want the key as a bare header instead of a Bearer token. */
  authHeader: 'bearer' | 'x-api-key' | 'none';
}

export const DEFAULT_LM: LMConfig = {
  baseUrl: 'https://api.openai.com/v1',
  apiKey: '',
  model: 'gpt-4o-mini',
  temperature: 0,
  maxTokens: 512,
  authHeader: 'bearer',
};

export const PRESETS: { id: string; label: string; baseUrl: string; model: string; note: string }[] = [
  { id: 'uvu', label: 'UVU AI Gateway', baseUrl: 'https://aigateway.uvu.edu/v1', model: 'gpt-4o-mini', note: 'Campus gateway. Uses your issued gateway key.' },
  { id: 'openai', label: 'OpenAI', baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o-mini', note: 'Standard OpenAI endpoint.' },
  { id: 'ollama', label: 'Ollama (local)', baseUrl: 'http://localhost:11434/v1', model: 'llama3.2', note: 'Runs entirely on this machine. No key needed.' },
  { id: 'lmstudio', label: 'LM Studio (local)', baseUrl: 'http://localhost:1234/v1', model: 'local-model', note: 'Point at whatever model you have loaded.' },
  { id: 'openrouter', label: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', model: 'openai/gpt-4o-mini', note: 'Many providers behind one key.' },
];

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LMCall {
  id: number;
  messages: ChatMessage[];
  response: string;
  promptTokens: number;
  completionTokens: number;
  latencyMs: number;
  error?: string;
  at: number;
  tag: string;
}

let callId = 0;

/** Rough token estimate for endpoints that do not return usage. */
export function estimateTokens(s: string): number {
  return Math.ceil(s.length / 4);
}

export class LMClient {
  cfg: LMConfig;
  calls: LMCall[] = [];
  onCall?: (c: LMCall) => void;
  /** Cache keyed on the exact request, so repeat evaluations stay cheap. */
  private cache = new Map<string, string>();
  useCache = true;

  constructor(cfg: LMConfig) {
    this.cfg = cfg;
  }

  get totalPromptTokens() {
    return this.calls.reduce((n, c) => n + c.promptTokens, 0);
  }
  get totalCompletionTokens() {
    return this.calls.reduce((n, c) => n + c.completionTokens, 0);
  }
  get errorCount() {
    return this.calls.filter((c) => c.error).length;
  }

  clearCache() {
    this.cache.clear();
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.cfg.apiKey && this.cfg.authHeader === 'bearer') {
      h.Authorization = `Bearer ${this.cfg.apiKey}`;
    } else if (this.cfg.apiKey && this.cfg.authHeader === 'x-api-key') {
      h['x-api-key'] = this.cfg.apiKey;
    }
    return h;
  }

  async complete(messages: ChatMessage[], tag = 'predict', signal?: AbortSignal): Promise<string> {
    const key = JSON.stringify([this.cfg.model, this.cfg.temperature, messages]);
    if (this.useCache && this.cache.has(key)) return this.cache.get(key)!;

    const t0 = performance.now();
    const url = `${this.cfg.baseUrl.replace(/\/+$/, '')}/chat/completions`;
    const call: LMCall = {
      id: ++callId,
      messages,
      response: '',
      promptTokens: 0,
      completionTokens: 0,
      latencyMs: 0,
      at: Date.now(),
      tag,
    };

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: this.headers(),
        signal,
        body: JSON.stringify({
          model: this.cfg.model,
          messages,
          temperature: this.cfg.temperature,
          max_tokens: this.cfg.maxTokens,
        }),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`HTTP ${res.status}: ${body.slice(0, 300)}`);
      }
      const json = await res.json();
      const text: string = json?.choices?.[0]?.message?.content ?? '';
      call.response = text;
      call.promptTokens = json?.usage?.prompt_tokens ?? estimateTokens(messages.map((m) => m.content).join('\n'));
      call.completionTokens = json?.usage?.completion_tokens ?? estimateTokens(text);
      call.latencyMs = performance.now() - t0;
      this.calls.push(call);
      this.onCall?.(call);
      if (this.useCache) this.cache.set(key, text);
      return text;
    } catch (err) {
      call.error = err instanceof Error ? err.message : String(err);
      call.latencyMs = performance.now() - t0;
      this.calls.push(call);
      this.onCall?.(call);
      throw err;
    }
  }

  /** Cheap reachability probe used by the Connect panel. */
  async test(): Promise<{ ok: boolean; detail: string; latencyMs: number }> {
    const t0 = performance.now();
    try {
      const out = await this.complete(
        [{ role: 'user', content: 'Reply with the single word: ready' }],
        'connection-test',
      );
      return { ok: true, detail: out.trim().slice(0, 80), latencyMs: performance.now() - t0 };
    } catch (err) {
      return {
        ok: false,
        detail: err instanceof Error ? err.message : String(err),
        latencyMs: performance.now() - t0,
      };
    }
  }
}
