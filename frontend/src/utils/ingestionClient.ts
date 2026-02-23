const ENV_CONVERTER_API_URL = import.meta.env.VITE_CONVERTER_API_URL as string | undefined;

function getDefaultBackendUrl(): string {
  if (typeof window !== "undefined" && window.location?.hostname === "localhost") return "http://localhost:10000";
  return "https://general-platform.onrender.com";
}

function buildCandidateBaseUrls(): string[] {
  if (ENV_CONVERTER_API_URL) return [ENV_CONVERTER_API_URL];
  return [getDefaultBackendUrl()];
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

/**
 * Fetch with automatic retry + exponential backoff for transient errors
 * (503 Service Unavailable, 429 Too Many Requests, network failures).
 * Render free-tier cold-starts and rate limits are the main culprits.
 */
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
          console.warn(`[fetchWithRetry] ${response.status} on ${url} — retry ${attempt + 1}/${maxRetries} in ${delay}ms`);
          await new Promise((r) => setTimeout(r, delay));
          continue;
        }
      }
      return response;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < maxRetries) {
        const delay = baseDelayMs * Math.pow(2, attempt);
        console.warn(`[fetchWithRetry] Network error on ${url} — retry ${attempt + 1}/${maxRetries} in ${delay}ms:`, lastError.message);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastError ?? new Error(`fetchWithRetry failed after ${maxRetries} retries`);
}

/** Small sleep helper to throttle request bursts */
export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** Wake up the Render service before bulk requests (fire-and-forget with retry) */
export async function warmUpIngestion(): Promise<void> {
  try {
    const base = await resolveIngestionBaseUrl();
    console.log("[DriveSync] Warming up ingestion service...");
    await fetchWithRetry(`${base}/health`, { method: "GET" }, 3, 2000);
    console.log("[DriveSync] Ingestion service is awake.");
  } catch {
    console.warn("[DriveSync] Warm-up failed — will retry on first real request.");
  }
}

async function resolveIngestionBaseUrl(): Promise<string> {
  if (resolvedBaseUrl) return resolvedBaseUrl;

  const candidates = buildCandidateBaseUrls();
  if (!candidates.length) {
    resolvedBaseUrl = getDefaultBackendUrl();
    return resolvedBaseUrl;
  }

  for (const base of candidates) {
    try {
      const res = await fetchWithTimeout(`${base}/health`, undefined, 800);
      if (res.ok) {
        resolvedBaseUrl = base;
        return base;
      }
    } catch {
      // try next
    }
  }

  resolvedBaseUrl = candidates[0];
  return resolvedBaseUrl;
}

export async function ingestClickUpList(
  listId: string,
  includeClosed = true
): Promise<{
  tasks: Array<{
    id: string;
    name: string;
    url?: string | null;
    status?: string | null;
    assignees?: string[];
    description?: string | null;
  }>;
}> {
  try {
    const baseUrl = await resolveIngestionBaseUrl();
    const response = await fetch(`${baseUrl}/ingest/clickup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ list_id: listId, include_closed: includeClosed }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(
      `ClickUp ingestion failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

export async function getClickUpLists(
  teamId: string
): Promise<{ lists: Array<{ id: string; name: string }> }> {
  try {
    const baseUrl = await resolveIngestionBaseUrl();
    const response = await fetch(`${baseUrl}/ingest/clickup/lists`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team_id: teamId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(
      `ClickUp list fetch failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

export async function ingestGoogleDrive(
  url: string,
  accessToken?: string | null
): Promise<{ title: string; content: string; raw_content: string; sourceType: string }> {
  try {
    const baseUrl = await resolveIngestionBaseUrl();
    const response = await fetch(`${baseUrl}/ingest/google-drive`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, access_token: accessToken || null }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(
      `Google Drive ingestion failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

// ─── Google Drive Folder-Sync Helpers ────────────────────────────────────────

export interface GDriveFolderEntry {
  id: string;
  name: string;
  modifiedTime?: string | null;
}

/** Drive file metadata from list-files API */
export interface GDriveFileEntry {
  id: string;
  name: string;
  mimeType: string;
  modifiedTime?: string | null;
  size?: string | null;
}

export async function listDriveFolders(
  accessToken: string,
  folderId: string
): Promise<GDriveFolderEntry[]> {
  try {
    const base = await resolveIngestionBaseUrl();
    const response = await fetchWithRetry(`${base}/gdrive/list-folders`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ access_token: accessToken, folder_id: folderId }),
    });
    if (!response.ok) {
      const text = await response.text();
      let detail = `HTTP ${response.status}`;
      try { const parsed = JSON.parse(text); detail = parsed?.detail || parsed?.error || text || detail; } catch { if (text) detail = `HTTP ${response.status}: ${text.slice(0, 200)}`; }
      throw new Error(detail);
    }
    const data = await response.json();
    return data.folders ?? [];
  } catch (error) {
    throw new Error(
      `Drive list-folders failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

export async function listDriveFiles(
  accessToken: string,
  folderId: string
): Promise<GDriveFileEntry[]> {
  try {
    const base = await resolveIngestionBaseUrl();
    const response = await fetchWithRetry(`${base}/gdrive/list-files`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ access_token: accessToken, folder_id: folderId }),
    });
    if (!response.ok) {
      const text = await response.text();
      let detail = `HTTP ${response.status}`;
      try { detail = JSON.parse(text)?.detail || detail; } catch {}
      throw new Error(detail);
    }
    const data = await response.json();
    return data.files ?? [];
  } catch (error) {
    throw new Error(
      `Drive list-files failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

export async function downloadDriveFile(
  accessToken: string,
  fileId: string,
  mimeType?: string,
  fileName?: string
): Promise<{ title: string; content: string; raw_content: string; sourceType: string; mimeType: string }> {
  try {
    const base = await resolveIngestionBaseUrl();
    const response = await fetchWithRetry(`${base}/gdrive/download-file`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        access_token: accessToken,
        file_id: fileId,
        mime_type: mimeType || null,
        file_name: fileName || null,
      }),
    });
    if (!response.ok) {
      const text = await response.text();
      let detail = `HTTP ${response.status}`;
      try { detail = JSON.parse(text)?.detail || detail; } catch {}
      throw new Error(detail);
    }
    return await response.json();
  } catch (error) {
    throw new Error(
      `Drive download-file failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}