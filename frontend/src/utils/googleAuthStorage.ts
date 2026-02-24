/**
 * Persist Google OAuth provider tokens so we can use them after Supabase session
 * no longer includes them (Supabase does not persist provider_token/provider_refresh_token).
 */
const STORAGE_KEY = "orbit_google_provider_tokens";
const ACCESS_TOKEN_MAX_AGE_MS = 50 * 60 * 1000; // 50 min (Google tokens ~1h)

export interface StoredGoogleTokens {
  access_token?: string | null;
  refresh_token?: string | null;
  saved_at: number;
}

export function saveGoogleProviderTokens(accessToken: string | null | undefined, refreshToken: string | null | undefined): void {
  if (typeof window === "undefined") return;
  try {
    const payload: StoredGoogleTokens = {
      access_token: accessToken || null,
      refresh_token: refreshToken || null,
      saved_at: Date.now(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // ignore
  }
}

export function getStoredGoogleTokens(): StoredGoogleTokens | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredGoogleTokens;
    return parsed && typeof parsed.saved_at === "number" ? parsed : null;
  } catch {
    return null;
  }
}

export function getStoredGoogleRefreshToken(): string | null {
  const t = getStoredGoogleTokens();
  return t?.refresh_token && typeof t.refresh_token === "string" ? t.refresh_token : null;
}

/** Use stored access_token only if it was saved recently (avoid using expired token). */
export function getStoredGoogleAccessToken(): string | null {
  const t = getStoredGoogleTokens();
  if (!t?.access_token || typeof t.access_token !== "string") return null;
  if (Date.now() - (t.saved_at || 0) > ACCESS_TOKEN_MAX_AGE_MS) return null;
  return t.access_token;
}

export function setStoredGoogleAccessToken(accessToken: string): void {
  const t = getStoredGoogleTokens();
  saveGoogleProviderTokens(accessToken, t?.refresh_token ?? null);
}

export function clearGoogleProviderTokens(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
