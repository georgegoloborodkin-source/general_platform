/**
 * Gmail API client – mirrors the patterns in ingestionClient.ts
 * Talks to the backend converter API's /gmail/* endpoints.
 */

const ENV_CONVERTER_API_URL = import.meta.env.VITE_CONVERTER_API_URL as string | undefined;

function getDefaultBackendUrl(): string {
  if (typeof window !== "undefined" && window.location?.hostname === "localhost") return "http://localhost:10000";
  return "https://general-platform.onrender.com";
}

let resolvedBaseUrl: string | null = null;

async function fetchWithTimeout(url: string, init?: RequestInit, timeoutMs = 800): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchWithRetry(
  url: string,
  init?: RequestInit,
  maxRetries = 4,
  baseDelayMs = 1500
): Promise<Response> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, init);
      if (response.status === 503 || response.status === 429) {
        const retryAfter = response.headers.get("Retry-After");
        const delay = retryAfter
          ? parseInt(retryAfter, 10) * 1000
          : baseDelayMs * Math.pow(2, attempt);
        if (attempt < maxRetries) {
          console.warn(`[Gmail] ${response.status} — retry ${attempt + 1}/${maxRetries} in ${delay}ms`);
          await new Promise((r) => setTimeout(r, delay));
          continue;
        }
      }
      return response;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < maxRetries) {
        const delay = baseDelayMs * Math.pow(2, attempt);
        console.warn(`[Gmail] Network error — retry ${attempt + 1}/${maxRetries} in ${delay}ms:`, lastError.message);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastError ?? new Error(`fetchWithRetry failed after ${maxRetries} retries`);
}

function buildCandidateBaseUrls(): string[] {
  if (ENV_CONVERTER_API_URL) return [ENV_CONVERTER_API_URL];
  return [getDefaultBackendUrl()];
}

async function resolveBaseUrl(): Promise<string> {
  if (resolvedBaseUrl) return resolvedBaseUrl;
  const candidates = buildCandidateBaseUrls();
  if (!candidates.length) {
    resolvedBaseUrl = getDefaultBackendUrl();
    return resolvedBaseUrl;
  }
  for (const base of candidates) {
    try {
      const res = await fetchWithTimeout(`${base}/health`, undefined, 800);
      if (res.ok) { resolvedBaseUrl = base; return base; }
    } catch { /* next */ }
  }
  resolvedBaseUrl = candidates[0];
  return resolvedBaseUrl;
}

// ─── Types ─────────────────────────────────────────────────────────────────

export interface GmailMessageSnippet {
  id: string;
  threadId: string;
  snippet?: string | null;
}

export interface GmailAttachmentMeta {
  id: string;
  filename: string;
  mimeType: string;
  size: number;
}

export interface GmailFullMessage {
  id: string;
  threadId: string;
  subject: string;
  sender: string;
  to: string[];
  cc: string[];
  date: string | null;
  body_text: string;
  body_html: string;
  labels: string[];
  attachments: GmailAttachmentMeta[];
}

export interface GmailIngestResult {
  title: string;
  content: string;
  raw_content: string;
  sourceType: string;
  email_from: string;
  email_to: string[];
  email_cc: string[];
  email_subject: string;
  email_date: string | null;
  gmail_thread_id: string;
  gmail_labels: string[];
  has_attachments: boolean;
  attachments: GmailAttachmentMeta[];
}

// ─── API Calls ─────────────────────────────────────────────────────────────

export async function gmailListMessages(
  accessToken: string,
  opts?: {
    query?: string;
    labelIds?: string[];
    maxResults?: number;
    pageToken?: string;
  }
): Promise<{ messages: GmailMessageSnippet[]; nextPageToken?: string; resultSizeEstimate: number }> {
  const base = await resolveBaseUrl();
  const response = await fetchWithRetry(`${base}/gmail/list-messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      access_token: accessToken,
      query: opts?.query ?? null,
      label_ids: opts?.labelIds ?? null,
      max_results: opts?.maxResults ?? 50,
      page_token: opts?.pageToken ?? null,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    let detail = `HTTP ${response.status}`;
    try { detail = JSON.parse(text)?.detail || detail; } catch { /* */ }
    throw new Error(`Gmail list-messages failed: ${detail}`);
  }

  const data = await response.json();
  return {
    messages: data.messages ?? [],
    nextPageToken: data.next_page_token ?? undefined,
    resultSizeEstimate: data.result_size_estimate ?? 0,
  };
}

export async function gmailGetMessage(
  accessToken: string,
  messageId: string
): Promise<GmailFullMessage> {
  const base = await resolveBaseUrl();
  const response = await fetchWithRetry(`${base}/gmail/get-message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ access_token: accessToken, message_id: messageId }),
  });

  if (!response.ok) {
    const text = await response.text();
    let detail = `HTTP ${response.status}`;
    try { detail = JSON.parse(text)?.detail || detail; } catch { /* */ }
    throw new Error(`Gmail get-message failed: ${detail}`);
  }

  return await response.json();
}

export async function gmailDownloadAttachment(
  accessToken: string,
  messageId: string,
  attachmentId: string
): Promise<{ data: string; filename: string; mimeType: string; size: number }> {
  const base = await resolveBaseUrl();
  const response = await fetchWithRetry(`${base}/gmail/download-attachment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      access_token: accessToken,
      message_id: messageId,
      attachment_id: attachmentId,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    let detail = `HTTP ${response.status}`;
    try { detail = JSON.parse(text)?.detail || detail; } catch { /* */ }
    throw new Error(`Gmail download-attachment failed: ${detail}`);
  }

  return await response.json();
}

export async function gmailIngestMessage(
  accessToken: string,
  messageId: string,
  extractAttachments = false
): Promise<GmailIngestResult> {
  const base = await resolveBaseUrl();
  const response = await fetchWithRetry(`${base}/ingest/gmail`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      access_token: accessToken,
      message_id: messageId,
      extract_attachments: extractAttachments,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    let detail = `HTTP ${response.status}`;
    try { detail = JSON.parse(text)?.detail || detail; } catch { /* */ }
    throw new Error(`Gmail ingest failed: ${detail}`);
  }

  return await response.json();
}
