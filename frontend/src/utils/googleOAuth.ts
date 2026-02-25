import { supabase } from "@/integrations/supabase/client";
import { clearMyToken404Cache } from "@/utils/ingestionClient";

const PRODUCTION_ORIGIN = "https://general-platform.vercel.app";
const BACKEND_ORIGIN = "https://general-platform.onrender.com";

export function getGoogleOAuthRedirectTo(): string {
  if (typeof window !== "undefined" && window.location?.hostname === "localhost") {
    return `${window.location.origin}/auth/callback`;
  }
  return `${PRODUCTION_ORIGIN}/auth/callback`;
}

/**
 * Start backend-driven Google Drive OAuth: backend stores tokens and returns them via GET /gdrive/my-token.
 * Redirects to backend -> Google -> back to frontend. No reliance on Supabase provider_token.
 */
export async function triggerGoogleOAuthForDrive(): Promise<void> {
  clearMyToken404Cache();

  let accessToken: string | undefined;

  // 1. Try cached session first (no network request)
  const { data: sessionData } = await supabase.auth.getSession();
  accessToken = sessionData?.session?.access_token;

  // 2. If no token (expired / 429'd earlier), do ONE explicit refresh
  if (!accessToken) {
    const { data: refreshData, error: refreshError } = await supabase.auth.refreshSession();
    if (refreshError) {
      const msg = refreshError.message ?? "";
      if (msg.includes("429") || msg.toLowerCase().includes("too many")) {
        throw new Error("Too many requests. Please wait 60 seconds and try again, or sign out and sign back in.");
      }
      throw new Error(
        "You must be signed in to connect Google Drive. If this keeps happening, sign out and sign back in."
      );
    }
    accessToken = refreshData?.session?.access_token;
  }

  if (!accessToken) {
    throw new Error(
      "You must be signed in to connect Google Drive. Sign out and sign back in to fix this."
    );
  }

  const res = await fetch(`${BACKEND_ORIGIN}/auth/google-drive/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ access_token: accessToken }),
  });
  if (!res.ok) {
    const text = await res.text();
    if (res.status === 429) {
      throw new Error("Too many requests. Please wait a minute and try again, or sign out and sign back in.");
    }
    throw new Error(text || `Backend returned ${res.status}`);
  }
  const json = await res.json();
  if (json.redirect_url) {
    window.location.href = json.redirect_url;
  }
}
