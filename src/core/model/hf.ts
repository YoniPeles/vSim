// Hugging Face Hub access from the browser (or Node). resolve/ URLs send CORS headers for any
// origin and accept Authorization; gated repos answer 401/403 with x-error-code GatedRepo.

import type { PresetSnapshot } from './presets.ts';

type Json = Record<string, unknown>;

// Minimal fetch surface, so the core stays free of DOM typings yet runs in browsers, workers and Node.
interface ResponseLike {
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  json(): Promise<unknown>;
}
export type FetchLike = (url: string, init?: { headers?: Record<string, string> }) => Promise<ResponseLike>;
const globalFetch: FetchLike = (url, init) => (globalThis as unknown as { fetch: FetchLike }).fetch(url, init);

export class HubError extends Error {
  readonly code: 'gated' | 'not-found' | 'network' | 'invalid';
  constructor(code: HubError['code'], message: string) {
    super(message);
    this.code = code;
  }
}

const HUB = 'https://huggingface.co';

async function fetchJson(url: string, token?: string, f: FetchLike = globalFetch): Promise<Json | undefined> {
  let res: ResponseLike;
  try {
    res = await f(url, token ? { headers: { Authorization: `Bearer ${token}` } } : undefined);
  } catch (e) {
    throw new HubError('network', `Couldn't reach huggingface.co (${(e as Error).message}).`);
  }
  if (res.status === 404) return undefined;
  if (res.status === 401 || res.status === 403) {
    const code = res.headers.get('x-error-code');
    throw new HubError(
      'gated',
      code === 'GatedRepo' || res.status === 403
        ? token
          ? 'This repo is gated and your token has not been granted access. Accept the license on huggingface.co, or pick a bundled preset.'
          : 'This repo is gated. Add a Hugging Face token with access, or pick a bundled preset.'
        : 'Hugging Face rejected the request (unauthorized).',
    );
  }
  if (!res.ok) throw new HubError('network', `huggingface.co answered ${res.status} for ${url}.`);
  return (await res.json()) as Json;
}

/** Fetch config.json (+ hf_quant_config.json + safetensors param counts) for a repo. */
export async function fetchSnapshot(repo: string, opts: { token?: string; revision?: string; fetch?: FetchLike } = {}): Promise<PresetSnapshot> {
  const id = repo.trim().replace(/^https?:\/\/huggingface\.co\//, '').replace(/\/$/, '');
  if (!/^[\w.-]+\/[\w.-]+$/.test(id)) throw new HubError('invalid', 'Enter a repo id like "Qwen/Qwen3-8B".');
  const rev = opts.revision ?? 'main';
  const config = await fetchJson(`${HUB}/${id}/resolve/${rev}/config.json`, opts.token, opts.fetch);
  if (!config) throw new HubError('not-found', `${id} has no config.json at revision ${rev}.`);
  const [hfQuantConfig, api] = await Promise.all([
    fetchJson(`${HUB}/${id}/resolve/${rev}/hf_quant_config.json`, opts.token, opts.fetch).catch(() => undefined),
    fetchJson(`${HUB}/api/models/${id}?expand[]=safetensors&expand[]=gated`, opts.token, opts.fetch).catch(() => undefined),
  ]);
  const st = api?.['safetensors'] as PresetSnapshot['safetensors'] | undefined;
  return {
    repo: id,
    source: id,
    label: id.split('/')[1] ?? id,
    fetchedAt: new Date().toISOString().slice(0, 10),
    gated: api ? api['gated'] !== false : false,
    ...(st ? { safetensors: st } : {}),
    config,
    ...(hfQuantConfig ? { hfQuantConfig } : {}),
  };
}

/** Text-generation model search for the repo picker (one API call). */
export async function searchModels(query: string, f: FetchLike = globalFetch): Promise<string[]> {
  if (query.trim().length < 2) return [];
  const url = `${HUB}/api/models?search=${encodeURIComponent(query)}&filter=text-generation&sort=downloads&limit=12`;
  const res = await f(url);
  if (!res.ok) return [];
  const list = (await res.json()) as { id: string }[];
  return list.map((m) => m.id);
}
