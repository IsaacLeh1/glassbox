/**
 * Hugging Face dataset loading, straight from the browser.
 *
 * Two public endpoints do all the work and both allow cross-origin requests,
 * so no key, no proxy and no server of our own is involved:
 *
 *   huggingface.co/api/datasets            -- search the hub
 *   datasets-server.huggingface.co/rows    -- read actual rows of a dataset
 *
 * Only public, non-gated datasets are reachable this way. Gated ones return an
 * error, which is surfaced to the user rather than swallowed.
 */

const HUB = 'https://huggingface.co/api';
const VIEWER = 'https://datasets-server.huggingface.co';

export interface HFDataset {
  id: string;
  author: string;
  likes: number;
  downloads: number;
  gated: boolean;
  tags: string[];
}

export interface HFSplit {
  config: string;
  split: string;
  numRows: number | null;
}

export interface HFRowsPage {
  columns: { name: string; type: string }[];
  rows: Record<string, unknown>[];
  numRowsTotal: number;
}

async function j<T>(url: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, { signal });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const body = await r.json();
      if (body?.error) detail = String(body.error);
    } catch {
      /* keep the status code */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export async function searchDatasets(query: string, limit = 20, signal?: AbortSignal): Promise<HFDataset[]> {
  const url = `${HUB}/datasets?search=${encodeURIComponent(query)}&limit=${limit}&sort=downloads&direction=-1`;
  const raw = await j<Record<string, unknown>[]>(url, signal);
  return raw.map((d) => ({
    id: String(d.id ?? ''),
    author: String(d.author ?? ''),
    likes: Number(d.likes ?? 0),
    downloads: Number(d.downloads ?? 0),
    gated: Boolean(d.gated),
    tags: Array.isArray(d.tags) ? (d.tags as string[]) : [],
  }));
}

export async function getSplits(dataset: string, signal?: AbortSignal): Promise<HFSplit[]> {
  const url = `${VIEWER}/splits?dataset=${encodeURIComponent(dataset)}`;
  const raw = await j<{ splits: { config: string; split: string; num_rows?: number }[] }>(url, signal);
  return (raw.splits ?? []).map((s) => ({
    config: s.config,
    split: s.split,
    numRows: s.num_rows ?? null,
  }));
}

/** The viewer caps a single request at 100 rows. */
export const MAX_ROWS_PER_REQUEST = 100;

export async function getRows(
  dataset: string,
  config: string,
  split: string,
  offset: number,
  length: number,
  signal?: AbortSignal,
): Promise<HFRowsPage> {
  const n = Math.min(length, MAX_ROWS_PER_REQUEST);
  const url =
    `${VIEWER}/rows?dataset=${encodeURIComponent(dataset)}` +
    `&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}` +
    `&offset=${offset}&length=${n}`;
  const raw = await j<{
    features: { name: string; type: { dtype?: string; _type?: string } }[];
    rows: { row: Record<string, unknown> }[];
    num_rows_total?: number;
  }>(url, signal);
  return {
    columns: (raw.features ?? []).map((f) => ({
      name: f.name,
      type: f.type?.dtype ?? f.type?._type ?? 'unknown',
    })),
    rows: (raw.rows ?? []).map((r) => r.row),
    numRowsTotal: raw.num_rows_total ?? 0,
  };
}

export interface LoadProgress {
  rowsLoaded: number;
  chars: number;
  requests: number;
}

/**
 * Pull rows until either the row budget or the character budget is reached.
 * The character budget matters most: it is what decides how long tokenizer
 * training and model training will take on this machine.
 */
export async function loadTextCorpus(
  opts: {
    dataset: string;
    config: string;
    split: string;
    column: string;
    maxRows: number;
    maxChars: number;
    joiner?: string;
  },
  onProgress?: (p: LoadProgress) => void,
  signal?: AbortSignal,
): Promise<{ text: string; rowsUsed: number; requests: number }> {
  const parts: string[] = [];
  let chars = 0;
  let offset = 0;
  let requests = 0;
  const joiner = opts.joiner ?? '\n';

  while (chars < opts.maxChars && offset < opts.maxRows) {
    if (signal?.aborted) break;
    const want = Math.min(MAX_ROWS_PER_REQUEST, opts.maxRows - offset);
    const page = await getRows(opts.dataset, opts.config, opts.split, offset, want, signal);
    requests++;
    if (page.rows.length === 0) break;
    for (const row of page.rows) {
      const raw = row[opts.column];
      const text = typeof raw === 'string' ? raw : raw == null ? '' : JSON.stringify(raw);
      if (!text) continue;
      parts.push(text);
      chars += text.length + joiner.length;
      if (chars >= opts.maxChars) break;
    }
    offset += page.rows.length;
    onProgress?.({ rowsLoaded: offset, chars, requests });
    if (page.rows.length < want) break;
  }

  return { text: parts.join(joiner).slice(0, opts.maxChars), rowsUsed: offset, requests };
}

/**
 * Starting points that are small, public, and genuinely useful for a tiny
 * model. Every entry here was checked against the live dataset viewer: the
 * config, split and column names below are the ones the API actually returns.
 * Datasets that still ship a loading script, and datasets that have been
 * renamed, are rejected by the viewer and are deliberately not listed.
 */
export const SUGGESTED: { id: string; config: string; split: string; column: string; note: string }[] = [
  {
    id: 'roneneldan/TinyStories',
    config: 'default',
    split: 'train',
    column: 'text',
    note: 'Short stories written with a deliberately small vocabulary, built for training tiny models. The best first choice here by a wide margin.',
  },
  {
    id: 'Trelis/tiny-shakespeare',
    config: 'default',
    split: 'train',
    column: 'Text',
    note: 'The classic character-level benchmark. Strong, regular structure that a small model picks up quickly.',
  },
  {
    id: 'Abirate/english_quotes',
    config: 'default',
    split: 'train',
    column: 'quote',
    note: 'Short quotations. Small enough to load in full and see quick progress.',
  },
  {
    id: 'fancyzhx/ag_news',
    config: 'default',
    split: 'train',
    column: 'text',
    note: 'News headlines and summaries. Fairly formulaic, which helps a small model find structure.',
  },
  {
    id: 'stanfordnlp/imdb',
    config: 'plain_text',
    split: 'train',
    column: 'text',
    note: 'Film reviews: long, messy, natural prose. Genuinely hard, and a good reality check.',
  },
];

/**
 * Guess which column holds the text worth training on, by reading one row and
 * picking the longest string field. Saves the reader from typing a column name
 * they have no way of knowing.
 */
export async function detectTextColumn(
  dataset: string,
  config: string,
  split: string,
  signal?: AbortSignal,
): Promise<{ column: string; columns: string[] }> {
  const page = await getRows(dataset, config, split, 0, 1, signal);
  const row = page.rows[0] ?? {};
  let best = page.columns[0]?.name ?? 'text';
  let bestLen = -1;
  for (const c of page.columns) {
    const v = row[c.name];
    if (typeof v === 'string' && v.length > bestLen) {
      bestLen = v.length;
      best = c.name;
    }
  }
  return { column: best, columns: page.columns.map((c) => c.name) };
}
